import os
import sys
import numpy as np
import torch
import random
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from IBRLLM import IBRLLM
from utils import TrafficDataset

class Tee:
    """Class to write to both console and file"""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout

# LoRA configuration
from peft import LoraConfig, get_peft_model, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"
TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"],
    'qwen': ["q_proj", "v_proj"]
}

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def peft_model(plm, plm_type, rank, print_trainable=False, task_type=TaskType.FEATURE_EXTRACTION):
    """Apply LoRA to PLM model"""
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=TARGET_MODULES[plm_type],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type
    )

    model = get_peft_model(plm, config)
    if print_trainable:
        print_trainable_parameters(model)
    return model

def print_trainable_parameters(model):
    """Print trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def save_model(args, model, save_dir):
    """Save model weights"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.rank > 0:
        model.plm.save_pretrained(save_dir)
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))

def load_model(args, model, model_dir):
    """Load model weights"""
    if args.rank > 0:
        model.plm.load_adapter(model_dir, adapter_name='default')
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model

def evaluate_test_f1(args, model, test_loader, test_df_original, le):
    """Evaluate model on test set and calculate F1 score using GR.csv"""
    model.eval()
    ordered_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            payload = batch['payload'].to(args.device).long()
            src_ip = batch['src_ip'].to(args.device).long()
            dst_ip = batch['dst_ip'].to(args.device).long()
            src_port = batch['src_port'].to(args.device).float()
            dst_port = batch['dst_port'].to(args.device).float()
            host = batch['host'].to(args.device).long()

            logits = model(payload, src_ip, dst_ip, src_port, dst_port, host)
            predictions = torch.argmax(logits, dim=1)
            ordered_predictions.extend(predictions.cpu().numpy().tolist())

    # Prepare test dataframe
    test_df_result = test_df_original.copy()
    test_df_result = test_df_result.dropna(subset=['payload']).reset_index(drop=True)
    
    if len(ordered_predictions) != len(test_df_result):
        ordered_predictions = ordered_predictions[:len(test_df_result)]
    
    predicted_labels = le.inverse_transform(ordered_predictions)
    test_df_result['label'] = predicted_labels

    # Load GR.csv and calculate metrics
    gr_path = 'data/GR.csv'
    if not os.path.exists(gr_path):
        print(f'Warning: {gr_path} not found. Cannot calculate F1 score.')
        return None

    gr_df = pd.read_csv(gr_path)
    key_columns = ['protocol', 'hex_src_ip', 'hex_dst_ip', 'src_port', 'dst_port', 'host', 'payload']
    
    if 'label' not in gr_df.columns:
        print('Warning: GR.csv missing label column. Cannot calculate F1 score.')
        return None

    # Build lookup dictionary
    gr_dict = {}
    for _, row in gr_df.iterrows():
        key = tuple(str(row[col]) for col in key_columns)
        if pd.notna(row['label']):
            gr_dict[key] = str(row['label'])

    # Match and extract labels
    true_labels = []
    pred_labels = []
    for _, row in test_df_result.iterrows():
        key = tuple(str(row[col]) for col in key_columns)
        if key in gr_dict:
            true_labels.append(gr_dict[key])
            pred_labels.append(row['label'])

    if len(true_labels) == 0:
        print('Warning: No matching rows found. Cannot calculate F1 score.')
        return None

    # Convert to numeric and calculate F1 score
    try:
        true_labels_numeric = le.transform(true_labels)
        pred_labels_numeric = le.transform(pred_labels)
    except ValueError as e:
        print(f'Warning: Label encoding error: {e}. Cannot calculate F1 score.')
        return None

    f1 = f1_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels_numeric, pred_labels_numeric)
    precision = precision_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)
    recall = recall_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)

    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'matched_rows': len(true_labels)
    }

def adapt(args, model, train_loader, val_loader, test_loader, test_df_original, le, checkpoint_dir, best_model_dir, num_classes, class_weights=None):
    """Training function"""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1))
    
    # Use weighted loss if class_weights is provided
    if class_weights is not None:
        class_weights = class_weights.to(args.device)
        loss_fn = CrossEntropyLoss(weight=class_weights)
        print(f'Using weighted CrossEntropyLoss with class weights on device {args.device}')
    else:
        loss_fn = CrossEntropyLoss()
        print('Using standard CrossEntropyLoss (no class weights)')

    use_val = (val_loader is not None)
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_test_f1 = 0.0
    total_train_losses = []

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(train_loader):
            payload = batch['payload'].to(args.device).long()
            src_ip = batch['src_ip'].to(args.device).long()
            dst_ip = batch['dst_ip'].to(args.device).long()
            src_port = batch['src_port'].to(args.device).float()
            dst_port = batch['dst_port'].to(args.device).float()
            host = batch['host'].to(args.device).long()
            labels = batch['label'].to(args.device).long()

            logits = model(payload, src_ip, dst_ip, src_port, dst_port, host)
            loss = loss_fn(logits, labels) / args.grad_accum_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            if ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            train_losses.append(loss.item() * args.grad_accum_steps)
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            if step % 100 == 0:
                print(f'Epoch {epoch}, Step {step}, Loss: {loss.item() * args.grad_accum_steps:.6f}')

        train_loss_mean = np.mean(train_losses)
        train_acc = train_correct / train_total
        total_train_losses.extend(train_losses)

        print('=' * 20, f'Training Epoch #{epoch}', '=' * 20)
        print(f'Train Loss: {train_loss_mean:.6f}, Train Accuracy: {train_acc:.4f}')

        # Evaluation and model saving
        if epoch % args.eval_per_epoch == 0:
            if use_val:
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        payload = batch['payload'].to(args.device).long()
                        src_ip = batch['src_ip'].to(args.device).long()
                        dst_ip = batch['dst_ip'].to(args.device).long()
                        src_port = batch['src_port'].to(args.device).float()
                        dst_port = batch['dst_port'].to(args.device).float()
                        host = batch['host'].to(args.device).long()
                        labels = batch['label'].to(args.device).long()

                        logits = model(payload, src_ip, dst_ip, src_port, dst_port, host)
                        predictions = torch.argmax(logits, dim=1)
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)

                val_acc = val_correct / val_total
                print('>' * 10, 'Validation Information')
                print(f'Val Accuracy: {val_acc:.4f}')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_model(args, model, best_model_dir)
                    print(f'Best model saved (Val Accuracy: {best_val_acc:.4f})')
            else:
                # Evaluate on test set using GR.csv
                print('>' * 10, 'Test Evaluation (using GR.csv)')
                test_metrics = evaluate_test_f1(args, model, test_loader, test_df_original, le)
                if test_metrics is not None:
                    test_f1 = test_metrics['f1']
                    print(f'Test F1 Score: {test_f1:.4f}')
                    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
                    print(f'Test Precision: {test_metrics["precision"]:.4f}')
                    print(f'Test Recall: {test_metrics["recall"]:.4f}')
                    print(f'Matched rows: {test_metrics["matched_rows"]}')

                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        save_model(args, model, best_model_dir)
                        print(f'Best model saved (Test F1 Score: {best_test_f1:.4f})')
                else:
                    # Fallback to train accuracy if GR.csv is not available
                    if train_acc > best_train_acc:
                        best_train_acc = train_acc
                        save_model(args, model, best_model_dir)
                        print(f'Best model saved (Train Accuracy: {best_train_acc:.4f})')

        # Save checkpoint
        if (epoch + 1) % args.save_checkpoint_per_epoch == 0:
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, str(epoch))
            save_model(args, model, checkpoint_dir_epoch)
            print(f'Checkpoint saved at: {checkpoint_dir_epoch}')

    # Save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')
    print(f'Training losses saved at: {train_losses_path}')

def test(args, model, test_loader, model_dir, le, test_df_original):
    """Testing function - saves predictions and evaluates using GR.csv"""
    model = load_model(args, model, model_dir)
    print(f'Load model from: {model_dir}')

    model.eval()
    ordered_predictions = []
    loss_fn = CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            payload = batch['payload'].to(args.device).long()
            src_ip = batch['src_ip'].to(args.device).long()
            dst_ip = batch['dst_ip'].to(args.device).long()
            src_port = batch['src_port'].to(args.device).float()
            dst_port = batch['dst_port'].to(args.device).float()
            host = batch['host'].to(args.device).long()

            logits = model(payload, src_ip, dst_ip, src_port, dst_port, host)
            predictions = torch.argmax(logits, dim=1)
            ordered_predictions.extend(predictions.cpu().numpy().tolist())

    # Save predictions to result.csv
    test_df_result = test_df_original.copy()
    test_df_result = test_df_result.dropna(subset=['payload']).reset_index(drop=True)
    
    if len(ordered_predictions) != len(test_df_result):
        print(f'Warning: Predictions ({len(ordered_predictions)}) != rows ({len(test_df_result)})')
        ordered_predictions = ordered_predictions[:len(test_df_result)]
    
    predicted_labels = le.inverse_transform(ordered_predictions)
    test_df_result['label'] = predicted_labels
    test_df_result.to_csv('data/result.csv', index=False)
    print(f'Predictions saved to data/result.csv ({len(predicted_labels)} rows)')

    # Load GR.csv and calculate metrics
    gr_path = 'data/GR.csv'
    if not os.path.exists(gr_path):
        print(f'Warning: {gr_path} not found. Skipping evaluation.')
        return {}

    print(f'Loading ground truth from {gr_path}...')
    gr_df = pd.read_csv(gr_path)
    key_columns = ['protocol', 'hex_src_ip', 'hex_dst_ip', 'src_port', 'dst_port', 'host', 'payload']
    
    if 'label' not in gr_df.columns:
        print('Warning: GR.csv missing label column. Skipping evaluation.')
        return {}

    # Build lookup dictionary
    gr_dict = {}
    for _, row in gr_df.iterrows():
        key = tuple(str(row[col]) for col in key_columns)
        if pd.notna(row['label']):
            gr_dict[key] = str(row['label'])

    # Match and extract labels
    true_labels = []
    pred_labels = []
    for _, row in test_df_result.iterrows():
        key = tuple(str(row[col]) for col in key_columns)
        if key in gr_dict:
            true_labels.append(gr_dict[key])
            pred_labels.append(row['label'])

    if len(true_labels) == 0:
        print('Warning: No matching rows found. Skipping evaluation.')
        return {}

    # Convert to numeric and calculate metrics
    try:
        true_labels_numeric = le.transform(true_labels)
        pred_labels_numeric = le.transform(pred_labels)
    except ValueError as e:
        print(f'Warning: Label encoding error: {e}. Some labels may not be in training set.')
        return {}

    accuracy = accuracy_score(true_labels_numeric, pred_labels_numeric)
    precision = precision_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)
    recall = recall_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)
    f1 = f1_score(true_labels_numeric, pred_labels_numeric, average='macro', zero_division=0)

    print('\n' + '=' * 60)
    print('Evaluation Results (using GR.csv as ground truth)')
    print('=' * 60)
    print(f'Matched rows: {len(true_labels)}/{len(test_df_result)}')
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print('=' * 60)
    print('\nDetailed Classification Report:')
    print(classification_report(true_labels_numeric, pred_labels_numeric, zero_division=0))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def run(args):
    set_random_seed(args.seed)

    # Prepare model path
    if args.model_path is None:
        plm_dir = args.plm_dir if hasattr(args, 'plm_dir') else '../../downloaded_plms'
        args.model_path = os.path.join(plm_dir, args.plm_type, args.plm_size)
    
    if args.llm_dim is None:
        print('LLM Dimensions Not Detected!')
        exit(-1)

    # Load data
    print(f'Loading training data from {args.train_file}...')
    train_df = pd.read_csv(args.train_file)
    train_df = train_df.dropna(subset=['payload'])

    print(f'Loading test data from {args.test_file}...')
    test_df_original = pd.read_csv(args.test_file)
    test_df = test_df_original.copy().dropna(subset=['payload'])

    # Fit LabelEncoder on training labels only (filter empty labels)
    le = LabelEncoder()
    train_df['label'] = train_df['label'].astype(str)
    valid_train_labels = train_df['label'][train_df['label'] != 'nan']
    le.fit(valid_train_labels)
    num_classes = len(le.classes_)
    print(f'Number of classes: {num_classes}')

    # Calculate class weights for balanced loss (if needed)
    class_weights = None
    if args.use_weighted_loss:
        print('Calculating balanced class weights...')
        # Count samples per class in training data
        label_counts = train_df['label'].value_counts()
        total_samples = len(train_df)
        
        # Calculate balanced weights: weight[i] = total_samples / (num_classes * class_i_samples)
        weights = []
        for i, class_name in enumerate(le.classes_):
            class_count = label_counts.get(class_name, 0)
            if class_count > 0:
                weight = total_samples / (num_classes * class_count)
            else:
                weight = 1.0  # Default weight if class has no samples
            weights.append(weight)
        
        # Convert to tensor and move to device
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f'Class weights calculated. Min: {class_weights.min():.4f}, Max: {class_weights.max():.4f}, Mean: {class_weights.mean():.4f}')
        print(f'Class weights: {class_weights.tolist()[:5]}... (showing first 5)')

    # Split training data
    if args.use_val:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
        train_dataset = TrafficDataset(train_df, le, max_payload_len=args.payload_max_len, max_host_len=args.host_max_len, has_label=True)
        val_dataset = TrafficDataset(val_df, le, max_payload_len=args.payload_max_len, max_host_len=args.host_max_len, has_label=True)
        print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_df)}')
    else:
        train_dataset = TrafficDataset(train_df, le, max_payload_len=args.payload_max_len, max_host_len=args.host_max_len, has_label=True)
        val_dataset = None
        print(f'Train: {len(train_dataset)}, Val: 0 (disabled), Test: {len(test_df)}')

    test_dataset = TrafficDataset(test_df, le, max_payload_len=args.payload_max_len, max_host_len=args.host_max_len, has_label=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    print(f'Creating IBRLLM model from {args.model_path}...')
    args.device = torch.device(args.device)
    args.num_classes = num_classes
    model = IBRLLM(args).to(args.device)

    if args.rank > 0:
        print(f'Applying LoRA with rank {args.rank}...')
        model.plm = peft_model(model.plm, args.plm_type, rank=args.rank, print_trainable=True)

    if args.frozen:
        for param in model.plm.parameters():
            param.requires_grad = False

    # Prepare directories
    model_name = f'ibrllm_rank_{args.rank}_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}'
    models_dir = os.path.join(args.output_dir, f'{args.plm_type}_{args.plm_size}', model_name)
    checkpoint_dir = os.path.join(models_dir, 'checkpoint')
    best_model_dir = os.path.join(models_dir, 'best_model')
    
    # Setup console logging
    console_log_path = os.path.join(models_dir, 'console.log')
    os.makedirs(models_dir, exist_ok=True)
    tee = Tee(console_log_path, mode='w')
    
    # Re-print arguments to log file
    print('Arguments:')
    pprint(vars(args))
    print('=' * 60)

    torch.backends.cudnn.benchmark = True

    try:
        if args.adapt:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(best_model_dir, exist_ok=True)
            print('Starting training...')
            adapt(args, model, train_loader, val_loader, test_loader, test_df_original, le, checkpoint_dir, best_model_dir, num_classes, class_weights)
        
        if args.test:
            model_dir = args.model_dir if args.model_dir else best_model_dir
            assert os.path.exists(model_dir), f'Model dir {model_dir} does not exist.'
            print(f'Testing model from {model_dir}...')
            test_results = test(args, model, test_loader, model_dir, le, test_df_original)
            if test_results:
                print('\nSummary:')
                print(f'  Accuracy: {test_results["accuracy"]:.4f}')
                print(f'  Precision: {test_results["precision"]:.4f}')
                print(f'  Recall: {test_results["recall"]:.4f}')
                print(f'  F1 Score: {test_results["f1"]:.4f}')
    finally:
        # Restore stdout and close log file
        tee.close()
        print(f'Console log saved to: {console_log_path}')

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Data settings
    parser.add_argument('--train-file', type=str, required=True, help='path to training CSV file')
    parser.add_argument('--test-file', type=str, required=True, help='path to test CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    
    # PLM settings
    parser.add_argument('--plm-type', type=str, default='llama', help='type of PLM')
    parser.add_argument('--plm-size', type=str, default='base', help='size of PLM')
    parser.add_argument('--model-path', type=str, default=None, help='path to PLM model')
    parser.add_argument('--plm-dir', type=str, default='../downloaded_plms', help='directory containing PLM models')
    parser.add_argument('--rank', type=int, default=128, help='LoRA rank')
    parser.add_argument('--frozen', action='store_true', help='freeze LLM parameters')
    
    # Model settings
    parser.add_argument('--llm-dim', type=int, default=None, help='LLM embedding dimension')
    parser.add_argument('--payload-max-len', type=int, default=256, help='maximum payload length')
    parser.add_argument('--host-max-len', type=int, default=64, help='maximum host length')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='warmup steps')
    parser.add_argument('--num-epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--eval-per-epoch', type=int, default=2, help='evaluation per epoch')
    parser.add_argument('--save-checkpoint-per-epoch', type=int, default=10, help='saving checkpoint per epoch')
    parser.add_argument('--grad-accum-steps', type=int, default=32, help='gradient accumulation steps')
    parser.add_argument('--use-val', action='store_true', help='use validation set')
    parser.add_argument('--use-weighted-loss', action='store_true', help='use balanced class weights in CrossEntropyLoss')
    
    # Other settings
    parser.add_argument('--adapt', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='device (cuda or cpu)')
    parser.add_argument('--model-dir', type=str, default=None, help='model weight dir for testing')
    parser.add_argument('--output-dir', type=str, default='models', help='output directory for models')
    
    parser.set_defaults(frozen=True)
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    
    assert args.save_checkpoint_per_epoch <= args.num_epochs, 'save_checkpoint_per_epoch should not exceed num_epochs'
    
    run(args)
