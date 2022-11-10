"""Main training script. Mind that wandb config has to be added manually."""

import wandb
import argparse
import tensorflow as tf
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
import models
import dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age recognition training script.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "img_folder_path",
        type=str,
        help="Path to folder with images grouped into classes.",
    )
    group.add_argument(
        "tfr_folder_path", type=str, help="Path to folder with TFRecord files."
    )
    parser.add_argument(
        "model", type=str, help="Model to use. Avaliable are cnn and cnn_vit."
    )
    parser.add_argument(
        "batch_size", type=int, help="Size of input batch for the model."
    )
    parser.add_argument(
        "n_classes", type=int, help="Number of classes being classified."
    )
    parser.add_argument(
        "epochs", type=int, help="Number of epochs to run training for."
    )
    parser.add_argument("img_w", type=int, help="Input image width.")
    parser.add_argument("img_h", type=int, help="Input image height.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--w_decay",
        type=float,
        default=0.001,
        help="Weight decay value for AdamW optimizer.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Percentage of training data to use for validation.",
    )
    parser.add_argument(
        "--one_hot",
        type=bool,
        default=True,
        help="Wheather to use one hot label encoding in datasets.",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Wheather to shuffle files in training dataset.",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed value for pseudo-random operations."
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=64,
        help="Embedding size for cnn_vit model. Has no effect if model is cnn only.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=6,
        help="Number of self-attention heads in cnn_vit model. Has no effect if model is cnn only.",
    )
    parser.add_argument(
        "--transformer_layers",
        type=int,
        default=6,
        help="Number of attention layers in cnn_vit model. Has no effect if model is cnn only.",
    )
    parser.add_argument(
        "--mlp_units",
        type=tuple[int, int],
        default=[2048, 1024],
        help="Number of neurons in dense layers in MLP layer of cnn_vit model. Has no effect if model is cnn only.",
    )
    parser.add_argument(
        "--finetune",
        type=bool,
        default=False,
        help="Wheather to unfreeze model backbone after initial training and perform another round of training.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=0.00001,
        help="Learning rate to use during finetuning.",
    )
    parser.add_argument(
        "--finetune_w_decay",
        type=float,
        default=0.001,
        help="Weight decay to use during finetuning.",
    )
    args = parser.parse_args()

    if args.seed:
        tf.random.set_seed(args.seed)

    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    strategy = tf.distribute.MirroredStrategy()

    AUTOTUNE = tf.data.AUTOTUNE
    INPUT_SHAPE = (args.img_w, args.img_h, 3)

    ## define your own logging names for Wandb
    CONFIG = dict(
        seed=args.seed,
        img_size=INPUT_SHAPE,
        num_classes=args.n_classes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    wandb.init(
        project="my_proj",
        group="my_group",
        name="my_name",
        job_type="my_job",
        config=CONFIG,
    )

    if args.img_folder_path is not None:
        train_ds, valid_ds, test_ds = dataloaders.create_image_dataset(
            dirpath=args.img_folder_path,
            batch_size=args.batch_size,
            img_w=INPUT_SHAPE[0],
            img_h=INPUT_SHAPE[1],
            val_split=args.val_split,
            one_hot=args.one_hot,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    elif args.tfr_folder_path is not None:
        train_ds, valid_ds, test_ds = dataloaders.create_tfr_dataset(
            dirpath=args.tfr_folder_path,
            batch_size=args.batch_size,
            img_w=INPUT_SHAPE[0],
            img_h=INPUT_SHAPE[1],
            val_split=args.val_split,
            one_hot=args.one_hot,
            shuffle=args.shuffle,
            seed=args.seed,
            num_classes=args.n_classes,
        )

    with strategy.scope():

        adamw = tfa.optimizers.AdamW(
            learning_rate=args.lr,
            weight_decay=args.w_decay,
        )
        optimizer = adamw
        if args.model == "cnn":
            model_age = models.create_cnn_network(
                input_shape=INPUT_SHAPE, num_classes=args.n_classes
            )
        elif args.model == "cnn_vit":
            model_age = models.create_cnn_vit_network(
                input_shape=INPUT_SHAPE,
                num_classes=args.n_classes,
                projection_dim=args.proj_dim,
                transformer_layers=args.transformer_layers,
                num_heads=args.n_heads,
                mlp_head_units=args.mlp_units,
            )
        else:
            print("Invalid model name specified!")

        model_age.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tfa.metrics.CohenKappa(args.n_classes),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tfa.metrics.F1Score(args.n_classes),
            ],
        )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True,
    )

    history = model_age.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=[WandbCallback(), early_stop],
    )

    if args.finetune:
        with strategy.scope():

            adamw = tfa.optimizers.AdamW(
                learning_rate=args.finetune_lr,
                weight_decay=args.finetune_w_decay,
            )
            optimizer = adamw

            model_age.trainable = True

            model_age.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(),
                    tfa.metrics.CohenKappa(args.n_classes),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tfa.metrics.F1Score(args.n_classes),
                ],
            )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True,
        )

        history = model_age.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.epochs,
            callbacks=[WandbCallback(), early_stop],
        )

    model_age.save("my_model")

    print("Test dataset evaluation:")
    eval_test = model_age.evaluate(train_ds)
    print("Valid dataset evaluation:")
    eval_valid = model_age.evaluate(valid_ds)
    print("Train dataset evaluation:")
    eval_train = model_age.evaluate(test_ds)
