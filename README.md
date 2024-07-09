# Tutorial - Deploy Llama3-TenyxChat-70B using Inferless
Tenyx has created [Llama-3-TenyxChat-70B](https://huggingface.co/tenyx/Llama3-TenyxChat-70B) by fine-tuning Llama3-70B. They leverage the Direct Preference Optimization (DPO) framework with the open-source AI feedback dataset UltraFeedback and incorporated their proprietary approach.
Llama-3-TenyxChat-70B was trained using eight A100s (80GB) for fifteen hours, with a training setup obtained from HuggingFaceH4 ([GitHub](https://github.com/huggingface/alignment-handbook)).


## TL;DR:
- Deployment of Llama3-TenyxChat-70B model using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/).
- You can expect an average latency of `20.14 sec` and throughput of `6.2 tokens per second`. This setup has an average cold start time of `28.32 sec`.
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Model import via GitHub with `input_schema.py` file.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
              "inputs": [
                {
                  "data": [
                    "What is AI?"
                  ],
                  "name": "prompt",
                  "shape": [
                    1
                  ],
                  "datatype": "BYTES"
                }
              ]
            }
            '
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
def infer(self, inputs):
    prompt = inputs["prompt"]
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self,args):
    self.model = None
```

For more information refer to the [Inferless docs](https://docs.inferless.com/).
