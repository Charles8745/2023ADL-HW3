
# ADL HW3: m11203404 陳旭霖
以下內容包含:
* Build up environment
* Download models
* Inference
* Training 
* Contact Information

## Build up environment
- Step1: Please create and activate your venv(python=3.10) first
- Step2: install torch
    ```
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
- Step3: install requirement.txt 
    ```
    pip install -r requirements.txt
    ```

## Download models
- Option1: Download by Shell Scripts
    ```
    bash ./download.sh
    ```
- Option2: Download by gDrive: 
    
    https://drive.google.com/file/d/1SkgUdM4GcFcmm3DiRwXdud1Wj9i1H2bA/view?usp=drive_link

## Inferance
- Required input json format:
    ```
    [    
        {
            "id": "d573ddd1-7bb9-468d-b906-e392223d9579",
            "instruction": "穿右穴而進，其下甚削，陷峽頗深，即下穿所入之峽也，以壁削路阻，不得達。\n幫我把這句話翻譯成現代文"
        },
        {
            "id": "e3c475ca-f2b2-4450-af6d-675e646c2488",
            "instruction": "闥活捉一豬，從頭咬至頂，放之地上，仍走。\n把這句話翻譯成現代文。"
        },
    ]

    ```
- Execute by Shell Scripts
    ```
    bash ./run.sh /path/to/Taiwan-LLaMa-folder /path/to/peft-folder /path/to/input.josn /path/to/output.json
    ```
## Training
- Required jsonl format:
    ```
    [
        {
            "id": "db63fb72-e211-4596-94a4-69617706f7ef",
            "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
            "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
        },
        {
            "id": "a48b0e8f-dc7a-4130-acc6-a91cc4a81bd1",
            "instruction": "沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文",
            "output": "後未旬，果見囚執。"
        },
    ]
    ```

- Train
    ```
    # peft model save at ./tmp
    python train.py --pre_trained_model ./pre-trained_llama_model --train_dataset_path ./input_data.json  --output_dir ./merge_model_folder
    ```


## Contact
- Email: charles77778888asd@gmail.com 
- linkedin: www.linkedin.com/in/旭霖-陳-b34102277





