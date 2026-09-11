import time
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    # Wrap dataloader with tqdm for batch-level progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Update progress bar with current batch loss and accuracy
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(y_pred_class == y).sum().item() / len(y_pred):.4f}"
        })

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0

    # Wrap dataloader with tqdm for batch-level progress bar
    progress_bar = tqdm(dataloader, desc="Testing", leave=False)
    with torch.inference_mode():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Update progress bar with current batch loss and accuracy
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(test_pred_labels == y).sum().item() / len(test_pred_labels):.4f}"
            })

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_path: str = "best_model.pth") -> Dict[str, List]:
    """Trains and tests a PyTorch model and saves the best model based on test accuracy."""
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    model.to(device)
    best_test_acc = 0.0
    best_model_wts = None

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)

        test_loss, test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, save_path)
            print(f"[INFO] New best model saved to {save_path} with test_acc: {best_test_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return results