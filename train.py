from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import pickle
from transformers import AutoImageProcessor, AutoTokenizer
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, DonutSwinModel
from torch import optim
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
import wandb
import argparse


class CompDataset(Dataset):
    def __init__(self, pickle_file,image_path_root):
        super().__init__()
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        self.components = []
        self.image_paths = []
        for k in data.keys():
            self.components.extend(data[k]['components'])
            self.image_paths.extend([ f"{os.path.join(image_path_root,k)}_page-{comp['page']}.png" for comp in data[k]['components']])

    def __len__(self):
        return len(self.components)

    def __getitem__(self, index):
        comp = self.components[index]
        img = Image.open(self.image_paths[index]).convert("RGB")
        bbox = comp['bbox']
        cropped_img = transforms.functional.crop(img,top=bbox[1],left=bbox[0],height=bbox[3],width=bbox[2])
        try:
            text = comp['text']
        except:
            text = comp['category']
        return (cropped_img, text, comp['category_id'])
    

def collate_fn(batch):
    images = [e[0] for e in batch]
    text = [e[1] for e in batch]
    labels = torch.tensor([e[2] for e in batch], dtype=torch.long)
    return (images, text, labels)

class FusionLayer(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=4):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.mhsa = nn.MultiheadAttention(self.visual_dim,num_heads,kdim=text_dim,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(self.visual_dim,self.visual_dim*2),nn.GELU(),nn.Linear(self.visual_dim*2,self.visual_dim))
        self.layer_norm = nn.LayerNorm(self.visual_dim)

    def forward(self, visual_embedding, textual_embedding):
        visual_embedding = visual_embedding.unsqueeze(1)
        textual_embedding = textual_embedding.unsqueeze(1)
        self.mhsa.to(visual_embedding.device)
        fused_embed,_ = self.mhsa(visual_embedding,textual_embedding,visual_embedding)
        normalize_fused_embed = self.layer_norm(fused_embed.squeeze(1)) + visual_embedding.squeeze(1)
        layer_embed = self.layer_norm(self.fc(normalize_fused_embed)) + normalize_fused_embed
        return layer_embed

class FusionAttentionModule(nn.Module):
    def __init__(self, visual_dim, text_dim, num_layers = 3):
        super().__init__()
        self.fusion_layers = nn.ModuleList([FusionLayer(visual_dim,text_dim) for _ in range(num_layers)])

    def forward(self, visual_embedding, textual_embedding):
        for layer in self.fusion_layers:
          visual_embedding = layer(visual_embedding,textual_embedding)
        return visual_embedding


class ComponentEncoder(nn.Module):
    def __init__(self,num_layers=3,freeze_visual=True, freeze_textual=True):
        super().__init__()
        self.visual_encoder = DonutSwinModel.from_pretrained("./donut_encoder")
        self.textual_encoder = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
        if freeze_visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
        if freeze_textual:
            for p in self.textual_encoder.parameters():
                p.requires_grad = False
        self.fusion_module = FusionAttentionModule(self.visual_encoder.config.hidden_size,self.textual_encoder.config.hidden_size,num_layers)

    def forward(self, image_inputs, text_inputs):
        visual_embedding = self.visual_encoder(**image_inputs).pooler_output
        textual_embedding = self.textual_encoder(**text_inputs).pooler_output
        fused_embedding = self.fusion_module(visual_embedding,textual_embedding)
        return fused_embedding

class ComponentDect(nn.Module):
    def __init__(self, num_classes,num_layers=3,freeze_visual=True, freeze_textual=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ComponentEncoder(num_layers,freeze_visual, freeze_textual)
        self.fc = nn.Linear(self.encoder.visual_encoder.config.hidden_size,num_classes)

    def forward(self, image_inputs, text_inputs):
        embed = self.encoder(image_inputs, text_inputs)
        pred = self.fc(embed)
        return pred

def focal_loss(pred, targets, alpha=0.25, gamma=2.0):
    log_prob = torch.log_softmax(pred,dim=1)
    one_hot_target = nn.functional.one_hot(targets,log_prob.shape[1])
    weight = alpha*torch.pow(1- log_prob.exp(),gamma)
    loss = torch.mean((-1*weight*log_prob)*one_hot_target)
    return loss

def train(model,train_dataloader, criterion, optimizer, image_processor, tokenizer, device):
    total_loss = 0
    model.train()
    for (images, texts, labels) in tqdm(train_dataloader):
        image_inputs = image_processor(images, return_tensors="pt").to(device)
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        labels = labels.to(device)
        pred = model(image_inputs,text_inputs)
        loss = criterion(pred,labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss/len(train_dataloader)

def val(model,val_dataloader, criterion, image_processor, tokenizer, device):
    total_loss = 0
    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for (images, texts, labels) in tqdm(val_dataloader):
            image_inputs = image_processor(images, return_tensors="pt").to(device)
            text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = labels.to(device)
            pred = model(image_inputs,text_inputs)
            loss = criterion(pred,labels)
            total_loss += loss.item()
            pred_classes = torch.argmax(pred,dim=1)
            predictions.append(pred_classes.detach().cpu())
            all_labels.append(labels.detach().cpu())
    predictions = torch.concat(predictions,dim=0)
    all_labels = torch.concat(all_labels,dim=0)
    f1 = f1_score(all_labels.numpy(), predictions.numpy())
    acc =accuracy_score(all_labels.numpy(), predictions.numpy())
    return loss/len(val_dataloader), f1, acc

def main():
    wandb.login(key="330d535f50c89464539700ab0fb6744d8043971b")
    with wandb.init(project="vrd-iu"):
        train_dataset = CompDataset('train_data.pkl','train/train')
        val_dataset = CompDataset('val_data.pkl','val/val')
        train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=True,collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset,batch_size=1,collate_fn=collate_fn)
        image_processor = AutoImageProcessor.from_pretrained("nielsr/donut-base")
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        model = ComponentDect(num_classes=25,num_layers=1,freeze_textual=True, freeze_visual=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model = model.to(device)
        wandb.watch(model, log="all")
        epochs = 50
        optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5,)
        best_f1_score = 0
        for epoch in tqdm(range(epochs)):
            val_loss, f1, acc = val(model,val_dataloader,focal_loss,image_processor,tokenizer,device)
            train_loss = train(model,train_dataloader,focal_loss,optimizer,image_processor,tokenizer,device)
            val_loss, f1, acc = val(model,val_dataloader,focal_loss,image_processor,tokenizer,device)
            print(f'Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss} F1 score: {f1} Accuracy: {acc}')
            if best_f1_score < f1:
                best_f1_score = f1
                torch.save(model.state_dict(),"comp_enc.pth")
                artifact = wandb.Artifact("comp-enc",type="model")
                artifact.add_file("comp_enc.pth")
                wandb.log_artifact(artifact)
            wandb.log({"train_loss":train_loss,"val_loss":val_loss,"accuracy":acc, "f1_score": f1,"lr":scheduler.get_last_lr()[0]})
            scheduler.step()

if __name__=="__main__":
    main()