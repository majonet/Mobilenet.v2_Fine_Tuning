import torch
import logging
from tqdm import tqdm
import os
import torch.nn.functional as F

def train(train_loader, val_loader, net, criterion, optimizer, device, epochs=20, save_path="/kaggle/working/"):
    net.to(device)
    best_val_loss = float("inf")  # Initialize best validation loss

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        

        for i, data in enumerate(train_loader):
            images, boxes, labels = data
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            optimizer.zero_grad()
            confidence, locations = net(images)

            image_width, image_height = 300,300
            gt_boxes_list = [
                b / torch.tensor([image_width, image_height, image_width, image_height], device=device)
                for b in boxes
            ]
            classification_loss = 0.0
            regression_loss = 0.0

            for i in range(len(labels)):
                   regression_loss_, classification_loss_= criterion(confidence[i], locations[i], labels[i], gt_boxes_list[i])
                   classification_loss+=classification_loss_
                   regression_loss+=regression_loss_
            # print("regression_loss",regression_loss)
            # classification_loss=model_train_loss(net)
            # print("classification_loss",classification_loss)
            loss = regression_loss+classification_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {running_loss:.4f}, Reg: {running_regression_loss:.4f}, Clf: {running_classification_loss:.4f}")
        print("----" * 20)

        # Validation after each epoch
        if epoch % 5 == 0 :
            net.eval()
            val_loss = 0.0
            val_reg_loss = 0.0
            val_clf_loss = 0.0

            with torch.no_grad():
                for data in val_loader:
                    images, boxes, labels = data
                    images = images.to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]

                    gt_boxes_list = [
                        b / torch.tensor([image_width, image_height, image_width, image_height], device=device)
                        for b in boxes
                    ]

                    confidence, locations = net(images)
                    for i in range(len(labels)):
                       regression_loss_, classification_loss_= criterion(confidence[i], locations[i], labels[i], gt_boxes_list[i])
                       classification_loss+=classification_loss_
                       regression_loss+=regression_loss_
                    loss = regression_loss + classification_loss

                    val_loss += loss.item()
                    val_reg_loss += regression_loss.item()
                    val_clf_loss += classification_loss.item()

            print(f"Validation Loss: {val_loss:.4f}, Reg: {val_reg_loss:.4f}, Clf: {val_clf_loss:.4f}")
            print("////" * 20)

            # Save best model
            # if val_loss < best_val_loss:
            print("ok")
            best_val_loss = val_loss
            torch.save(net.state_dict(), f"/kaggle/working/train_1/model_{epoch+110}.pth")
            print(f"model saved with val_loss: {best_val_loss:.4f}")
            # model_path=f"/kaggle/working/train_1/model_{val_loss}.pth"
            print("on data test")
            # model_test_data(net)
            print("on data train")
            # model_train_data(m)
            