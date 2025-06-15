import matplotlib.pyplot as plt
import glob


def load_loss_logs(log_dir):
    log_files = sorted(glob.glob(f"{log_dir}/events.*"))
    if not log_files:
        print("Логи не найдены")
        return [], []

    from tensorboard.backend.event_processing import event_accumulator

    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    steps = []
    train_losses = []
    eval_losses = []

    if "train/loss" in event_acc.Tags()["scalars"]:
        for event in event_acc.Scalars("train/loss"):
            steps.append(event.step)
            train_losses.append(event.value)

    if "eval/loss" in event_acc.Tags()["scalars"]:
        eval_steps = []
        for event in event_acc.Scalars("eval/loss"):
            eval_steps.append(event.step)
            eval_losses.append(event.value)

    return steps, train_losses, eval_steps, eval_losses


steps, train_losses, eval_steps, eval_losses = [], [], [], []

try:
    steps, train_losses, eval_steps, eval_losses = load_loss_logs("logs")
except Exception as e:
    print("Ошибка при чтении логов:", e)

# Построение графика
plt.figure(figsize=(10, 6))
if train_losses:
    plt.plot(steps, train_losses, label="Train Loss", marker='o')
if eval_losses:
    plt.plot(eval_steps, eval_losses, label="Eval Loss", marker='x')

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
