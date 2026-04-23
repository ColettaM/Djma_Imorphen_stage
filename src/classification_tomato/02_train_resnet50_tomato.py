"""
train_resnet50_tomato.py
----------------
Ce script correspond à une PREMIÈRE EXPÉRIENCE de classification des maladies
de la tomate avec ResNet50 dans le cadre du pipeline CNN.

Objectif de cette première expérience :
- établir un baseline solide et reproductible ;
- entraîner un modèle de classification multi-classes sur la base tomato_final ;
- conserver un protocole d’évaluation propre avec validation et test inchangés ;
- sauvegarder proprement les métriques, l’historique d’apprentissage et le mapping
  des classes pour exploitation ultérieure dans le mémoire ;
- ajouter un suivi MLflow pour tracer les paramètres, métriques, artefacts et modèles.

Choix méthodologiques de cette première version :
- architecture : ResNet50 pré-entraîné sur ImageNet ;
- fine-tuning en 2 phases :
    1) entraînement de la tête de classification seulement ;
    2) fine-tuning partiel ou total du backbone ;
- critère principal de sélection : macro-F1 sur l’ensemble de validation ;
- early stopping pour limiter le surapprentissage ;
- augmentation ONLINE légère uniquement sur train ;
- pas d’utilisation des class_weights dans cette première expérience.

Pourquoi ne pas utiliser de class_weights ici ?
-----------------------------------------------
Le dossier train a déjà été modifié en amont pour réduire le déséquilibre :
- sous-échantillonnage de certaines classes majoritaires ;
- augmentation ciblée de certaines classes minoritaires.

Dans ce contexte, ajouter en plus des class_weights dès cette première
expérience risquerait de sur-corriger le déséquilibre et de rendre
l’interprétation des résultats moins claire. Le choix retenu ici est donc
de commencer par un baseline sans class_weights, puis d’évaluer cette option
dans une expérience comparative ultérieure si nécessaire.

Pourquoi une augmentation légère seulement ?
--------------------------------------------
Le dossier train a déjà été modifié en amont :
- sous-échantillonnage de certaines classes majoritaires ;
- augmentation ciblée de certaines classes minoritaires.

Il n’est donc pas judicieux, dans cette première expérience, de réappliquer une
augmentation trop agressive dans le DataLoader. Ici, l’augmentation online est
volontairement modérée et sert surtout à améliorer la robustesse du modèle,
pas à rééquilibrer de nouveau les classes.

Exemple d’exécution :
---------------------
python train_resnet50_tomato.py \
    --data_dir /chemin/vers/tomato_final \
    --output_dir /chemin/vers/results_resnet50 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs_head 5 \
    --epochs_ft 15 \
    --lr_head 1e-3 \
    --lr_ft 1e-4 \
    --unfreeze layer4 \
    --mlflow_experiment Tomato_ResNet50_Baseline \
    --mlflow_run_name resnet50_layer4_seed42
"""

import os
import json
import copy
import time
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import mlflow
import mlflow.pytorch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Permet de charger certaines images partiellement corrompues/tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# 1. OUTILS GÉNÉRAUX
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """
    Fixe les graines aléatoires pour rendre les expériences plus reproductibles.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data, save_path: str) -> None:
    """
    Sauvegarde un objet Python au format JSON avec indentation.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_epoch_metrics_to_mlflow(
    epoch,
    train_loss,
    train_acc,
    train_f1,
    val_loss,
    val_acc,
    val_f1,
    lr,
    phase,
):
    """
    Log les métriques d'une epoch dans MLflow.
    """
    mlflow.log_metrics(
        {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": float(lr),
        },
        step=epoch,
    )
    mlflow.set_tag("current_phase", phase)


# =============================================================================
# 2. TRANSFORMS
# =============================================================================
def get_transforms(img_size: int = 224):
    """
    Définit les transformations appliquées aux images.

    Choix pour cette première expérience :
    - augmentation légère sur train seulement ;
    - preprocessing standard sur val/test.

    Remarque :
    Le train ayant déjà été rééquilibré en amont, on évite ici les
    transformations trop agressives.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


# =============================================================================
# 3. CHARGEMENT DES DONNÉES
# =============================================================================
def build_dataloaders(data_dir: str, batch_size: int, num_workers: int, img_size: int):
    """
    Charge les datasets train / val / test à partir de l’arborescence
    compatible ImageFolder.

    Retourne :
    - les datasets ;
    - les DataLoaders ;
    - les noms de classes ;
    - le mapping class_to_idx.
    """
    train_tf, eval_tf = get_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train_dir introuvable : {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val_dir introuvable : {val_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir introuvable : {test_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    if (
        train_dataset.classes != val_dataset.classes
        or train_dataset.classes != test_dataset.classes
    ):
        raise ValueError("Les classes de train/val/test ne sont pas cohérentes.")

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_names,
        class_to_idx,
        idx_to_class,
    )


# =============================================================================
# 4. MODÈLE
# =============================================================================
def build_model(num_classes: int, dropout: float = 0.3):
    """
    Construit un ResNet50 pré-entraîné sur ImageNet et remplace la couche finale
    pour l’adapter au nombre de classes du problème.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def freeze_all_backbone(model: nn.Module):
    """
    Gèle tout le backbone et laisse entraînable seulement la tête de classification.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_last_block(model: nn.Module):
    """
    Dégèle uniquement le dernier bloc convolutionnel (layer4)
    ainsi que la tête de classification.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module):
    """
    Dégèle toutes les couches du modèle.
    """
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# 5. BOUCLES D'ENTRAÎNEMENT / ÉVALUATION
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Exécute une epoch d'entraînement.
    Retourne :
    - loss moyenne
    - accuracy
    - macro-F1
    """
    model.train()

    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sur un loader donné.
    Retourne :
    - loss moyenne
    - accuracy
    - macro-F1
    - y_true
    - y_pred
    """
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, np.array(all_targets), np.array(all_preds)


def run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    n_epochs,
    patience,
    ckpt_path,
    history,
    start_epoch_idx=0,
    phase_name="head",
    best_val_f1_global=-1.0,
):
    """
    Lance une phase d'entraînement.
    Le meilleur modèle global est sélectionné selon le macro-F1 de validation.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1_local = -1.0
    patience_counter = 0

    for local_epoch in range(n_epochs):
        epoch = start_epoch_idx + local_epoch + 1
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["phase"].append(phase_name)
        history["epoch"].append(epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:02d}] "
            f"phase={phase_name} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"time={time.time() - t0:.1f}s"
        )

        log_epoch_metrics_to_mlflow(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            train_f1=train_f1,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            lr=current_lr,
            phase=phase_name,
        )

        if val_f1 > best_val_f1_local:
            best_val_f1_local = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

        if val_f1 > best_val_f1_global:
            best_val_f1_global = val_f1
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0

            mlflow.log_metric(
                "best_val_macro_f1_so_far", float(best_val_f1_global), step=epoch
            )
            mlflow.set_tag("best_ckpt_updated", "true")

            print(
                f"  -> nouveau meilleur modèle global sauvegardé (val_macro_f1={best_val_f1_global:.4f})"
            )
        else:
            patience_counter += 1
            mlflow.set_tag("best_ckpt_updated", "false")
            print(f"  -> patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("  -> early stopping déclenché")
            break

    model.load_state_dict(best_model_wts)
    return model, best_val_f1_local, best_val_f1_global, history


# =============================================================================
# 6. VISUALISATIONS ET SAUVEGARDES
# =============================================================================
def save_history_plot(history, save_path):
    """
    Sauvegarde les courbes d’apprentissage.
    """
    epochs = history["epoch"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_f1"], marker="o", label="train_macro_f1")
    plt.plot(epochs, history["val_f1"], marker="o", label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Courbes de macro-F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Sauvegarde la matrice de confusion sur l’ensemble de test.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Matrice de confusion - Test")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_history_raw(history, output_dir):
    """
    Sauvegarde l’historique brut en JSON et CSV.
    """
    history_json_path = os.path.join(output_dir, "history.json")
    history_csv_path = os.path.join(output_dir, "history.csv")

    save_json(history, history_json_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8")


# =============================================================================
# 7. FONCTION PRINCIPALE
# =============================================================================
def main(args):
    """
    Fonction principale :
    - préparation des données
    - construction du modèle
    - entraînement en 2 phases
    - évaluation finale sur test
    - sauvegarde complète des résultats
    - suivi MLflow
    """
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.enabled = False
        print("cuDNN enabled:", torch.backends.cudnn.enabled)

        (
            train_dataset,
            val_dataset,
            test_dataset,
            train_loader,
            val_loader,
            test_loader,
            class_names,
            class_to_idx,
            idx_to_class,
        ) = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
        )

        print("=" * 80)
        print("RÉSUMÉ DE L’EXPÉRIENCE")
        print("=" * 80)
        print(f"Device                 : {device}")
        print(f"Data directory         : {args.data_dir}")
        print(f"Output directory       : {args.output_dir}")
        print(f"Nombre de classes      : {len(class_names)}")
        print(f"Train size             : {len(train_dataset)}")
        print(f"Validation size        : {len(val_dataset)}")
        print(f"Test size              : {len(test_dataset)}")
        print(f"Classes                : {class_names}")
        print("=" * 80)

        save_json(class_names, os.path.join(args.output_dir, "class_names.json"))
        save_json(class_to_idx, os.path.join(args.output_dir, "class_to_idx.json"))
        save_json(idx_to_class, os.path.join(args.output_dir, "idx_to_class.json"))

        config = vars(args)
        save_json(config, os.path.join(args.output_dir, "config.json"))

        mlflow.log_params(
            {
                "data_dir": args.data_dir,
                "output_dir": args.output_dir,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "epochs_head": args.epochs_head,
                "epochs_ft": args.epochs_ft,
                "lr_head": args.lr_head,
                "lr_ft": args.lr_ft,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "patience": args.patience,
                "unfreeze": args.unfreeze,
                "seed": args.seed,
                "num_classes": len(class_names),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "architecture": "resnet50",
                "pretrained_weights": "IMAGENET1K_V2",
                "selection_metric": "val_macro_f1",
                "class_weights_used": False,
                "training_protocol": "2_phases_head_then_finetuning",
            }
        )

        mlflow.set_tags(
            {
                "project": "tomato_disease_classification",
                "model_family": "ResNet50",
                "dataset": "tomato_final",
                "task": "multiclass_classification",
                "baseline_type": "first_experiment",
            }
        )

        mlflow.log_text("\n".join(class_names), "artifacts/class_names.txt")

        model = build_model(num_classes=len(class_names), dropout=args.dropout).to(
            device
        )

        criterion = nn.CrossEntropyLoss()

        history = {
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
        }

        best_ckpt_path = os.path.join(args.output_dir, "best_resnet50_tomato.pth")
        last_ckpt_path = os.path.join(args.output_dir, "last_resnet50_tomato.pth")

        best_val_f1_global = -1.0

        print("\n=== PHASE 1 : entraînement de la tête de classification ===")
        freeze_all_backbone(model)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        model, best_val_f1_head, best_val_f1_global, history = run_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            n_epochs=args.epochs_head,
            patience=args.patience,
            ckpt_path=best_ckpt_path,
            history=history,
            start_epoch_idx=0,
            phase_name="head",
            best_val_f1_global=best_val_f1_global,
        )

        mlflow.log_metric("best_val_macro_f1_head", float(best_val_f1_head))

        print("\n=== PHASE 2 : fine-tuning ===")

        if args.unfreeze == "layer4":
            unfreeze_last_block(model)
            print("Mode fine-tuning : layer4 + fc")
        elif args.unfreeze == "all":
            unfreeze_all(model)
            print("Mode fine-tuning : tout le réseau")
        else:
            raise ValueError("La valeur de --unfreeze doit être 'layer4' ou 'all'.")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_ft,
            weight_decay=args.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        model, best_val_f1_ft, best_val_f1_global, history = run_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            n_epochs=args.epochs_ft,
            patience=args.patience,
            ckpt_path=best_ckpt_path,
            history=history,
            start_epoch_idx=len(history["train_loss"]),
            phase_name="finetuning",
            best_val_f1_global=best_val_f1_global,
        )

        mlflow.log_metric("best_val_macro_f1_finetuning", float(best_val_f1_ft))
        mlflow.log_metric("best_val_macro_f1_global", float(best_val_f1_global))

        print("\n=== FIN DE L’ENTRAÎNEMENT ===")
        print(f"Meilleur macro-F1 validation global : {best_val_f1_global:.4f}")

        torch.save(model.state_dict(), last_ckpt_path)

        save_history_raw(history, args.output_dir)

        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        print("\n=== ÉVALUATION FINALE SUR TEST ===")
        test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        test_report = classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        )

        print(f"Test loss      : {test_loss:.4f}")
        print(f"Test accuracy  : {test_acc:.4f}")
        print(f"Test macro-F1  : {test_f1:.4f}")
        print("\nRapport de classification :")
        print(test_report)

        metrics_test = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "best_val_macro_f1_global": float(best_val_f1_global),
            "best_val_macro_f1_head": float(best_val_f1_head),
            "best_val_macro_f1_finetuning": float(best_val_f1_ft),
        }
        save_json(metrics_test, os.path.join(args.output_dir, "metrics_test.json"))

        with open(
            os.path.join(args.output_dir, "classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(test_report)

        save_history_plot(history, os.path.join(args.output_dir, "learning_curves.png"))

        save_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            os.path.join(args.output_dir, "confusion_matrix_test.png"),
        )

        mlflow.log_metrics(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "test_macro_f1": float(test_f1),
            }
        )

        mlflow.log_artifact(
            os.path.join(args.output_dir, "config.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "class_names.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "class_to_idx.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "idx_to_class.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "history.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "history.csv"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "metrics_test.json"), artifact_path="results"
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "classification_report.txt"),
            artifact_path="results",
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "learning_curves.png"),
            artifact_path="results",
        )
        mlflow.log_artifact(
            os.path.join(args.output_dir, "confusion_matrix_test.png"),
            artifact_path="results",
        )
        mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
        mlflow.log_artifact(last_ckpt_path, artifact_path="checkpoints")

        # Log du modèle entraîné
        mlflow.pytorch.log_model(model, artifact_path="model")

        print("\n=== FICHIERS SAUVEGARDÉS ===")
        print("- config.json")
        print("- class_names.json")
        print("- class_to_idx.json")
        print("- idx_to_class.json")
        print("- history.json")
        print("- history.csv")
        print("- metrics_test.json")
        print("- classification_report.txt")
        print("- learning_curves.png")
        print("- confusion_matrix_test.png")
        print("- best_resnet50_tomato.pth")
        print("- last_resnet50_tomato.pth")
        print(f"\nRésultats disponibles dans : {args.output_dir}")


# =============================================================================
# 8. ARGUMENTS CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Première expérience ResNet50 pour la classification des maladies de la tomate"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Chemin vers le dossier tomato_final contenant train/ val/ test/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_resnet50_tomato",
        help="Dossier de sortie pour sauvegarder modèles, métriques et figures",
    )

    parser.add_argument(
        "--img_size", type=int, default=224, help="Taille des images après crop"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Nombre de workers pour les DataLoaders",
    )

    parser.add_argument(
        "--epochs_head",
        type=int,
        default=5,
        help="Nombre d'epochs pour la phase 1 (tête de classification)",
    )
    parser.add_argument(
        "--epochs_ft",
        type=int,
        default=15,
        help="Nombre d'epochs pour la phase 2 (fine-tuning)",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3,
        help="Learning rate pour la tête de classification",
    )
    parser.add_argument(
        "--lr_ft", type=float, default=1e-4, help="Learning rate pour le fine-tuning"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay pour AdamW"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout appliqué avant la couche finale",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience pour l'early stopping"
    )
    parser.add_argument(
        "--unfreeze",
        type=str,
        default="layer4",
        choices=["layer4", "all"],
        help="Choix du fine-tuning : dernier bloc seulement ou tout le réseau",
    )

    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")

    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="Tomato_ResNet50_Baseline",
        help="Nom de l'expérience MLflow",
    )
    parser.add_argument(
        "--mlflow_run_name", type=str, default=None, help="Nom du run MLflow"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="Tracking URI MLflow (ex: http://localhost:5000)",
    )

    args = parser.parse_args()
    main(args)
