import torch

def main():
    print("torch:", torch.__version__)
    print("torch.version.cuda:", getattr(torch.version, 'cuda', None))
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"[{i}]", torch.cuda.get_device_name(i))

if __name__ == "__main__":
    main()
