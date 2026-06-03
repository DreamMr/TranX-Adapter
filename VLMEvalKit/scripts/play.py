from vlmeval.config import supported_VLM
import traceback

if __name__ == '__main__':
    model_path = r'DreamMr/TranXAdapter-Qwen3VL2B-RRDataset'
    print(model_path)
    model = supported_VLM['Qwen3VL_NPR'](verbose=True, model_path=model_path)
    fake_image_path = r'./fake.jpg'
    inp="Determine whether this image is a result of fake or something real. Just answer \"Real\" or \"Fake\"."
    # test fake
    ret = model.generate([dict(type='image', value=fake_image_path),
                            dict(type='text',value=inp)])
    print(f"Label: fake, Pred: {ret}")
