import torch

from attention_unet_moe_sar_model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        in_channels_optical=3,
        in_channels_sar=1,
        out_channels=2,
        num_experts=3,
        dropout_p=0.3,
        input_size=(256, 256),
        device=device,
    )

    model.eval()

    optical = torch.randn(4, 3, 256, 256).to(device)
    sar = torch.randn(4, 1, 256, 256).to(device)

    with torch.no_grad():
        seg_out, expert_outputs = model(optical, sar)

    print("Device:", device)
    print("Segmentation Output Shape:", seg_out.shape)
    print("Expert Outputs Shape:", expert_outputs.shape)


if __name__ == "__main__":
    main()
