import torch
from tqdm import tqdm, trange


def train(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    n_epochs: int,
    lr: float,
) -> None:
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = (
        [param for param in model.parameters() if param.requires_grad]
    )
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    n_epochs = trange(n_epochs)
    
    for epoch in n_epochs:

        train_dl = tqdm(train_dl)
        train_loss = 0.0
        train_loss_sum = 0.0
        train_accuracy = 0.0
        n_train_samples = 0
        n_train_accurate_samples = 0

        for image_batch, target_batch in train_dl:
            image_batch = image_batch.cuda()
            target_batch = target_batch.cuda()
            batch_size, *_ = image_batch.shape

            model.train()
            output_batch = model(image_batch)
            batch_loss = loss_fn(output_batch, target_batch)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            n_train_samples += batch_size
            n_train_accurate_samples += (
                output_batch.argmax(dim=1).eq(target_batch).sum().item()
            )
            train_loss_sum += batch_loss.item() * batch_size
            train_loss = train_loss_sum / n_train_samples
            train_accuracy = n_train_accurate_samples / n_train_samples
            train_dl.set_description(
                f'train loss: {train_loss:.4f}, ' 
                f'train accuracy: {train_accuracy:.4f}'
            )

        val_dl = tqdm(val_dl)
        val_loss = 0.0
        val_loss_sum = 0.0
        val_accuracy = 0.0
        n_val_samples = 0
        n_val_accurate_samples = 0

        for image_batch, target_batch in val_dl:
            image_batch = image_batch.cuda()
            target_batch = target_batch.cuda()
            batch_size, *_ = image_batch.shape

            model.eval()
            output_batch = model(image_batch)
            with torch.no_grad():
                batch_loss = loss_fn(output_batch, target_batch)

            n_val_samples += batch_size
            n_val_accurate_samples += (
                output_batch.argmax(dim=1).eq(target_batch).sum().item()
            )
            val_loss_sum += batch_loss.item() * batch_size
            val_loss = val_loss_sum / n_val_samples
            val_accuracy = n_val_accurate_samples / n_val_samples
            val_dl.set_description(
                f'val loss: {val_loss:.4f}, '
                f'val accuracy: {val_accuracy:.4f}'
            )

        n_epochs.set_description(
            f'train loss: {val_loss:.4f}, train accuracy: {train_accuracy:.4f} | '
            f'val loss: {val_loss:.4f}, val accuracyy: {val_accuracy:.4f}'
        )
