from ...smp import *
import os
from sklearn import metrics
from ...utils import track_progress_rich

EXTRACTION_FAKE_OR_REAL = """
Based on the information provided by the agent, determine whether the agent considers the image to be **real** or **fake**. Please respond with **"real"** or **"fake"** only. If the answer says things like refuse to answer, I'm sorry cannot help, etc., output Unknown.\n
Example 1:\n
Agent's response: "The image you've provided appears to be a digital artwork or manipulation rather than a photograph of a real scene."
Answer: fake\n
Example 2:\n
Agent's response: "The image you've provided appears to be a comic strip or a sequence of panels from a comic book or graphic novel. The style and presentation suggest it is a work of fiction, intended to be humorous or to tell a story. The authenticity of the image is that it is a piece of creative artwork, not a photograph of real events."
Answer: fake\n
Example 3:\n
Agent's response: "The image appears to be a real photograph of a person playing tennis. There are no obvious signs of manipulation or digital alteration that would indicate the image is fake. The person is holding a tennis racket and is in a typical tennis stance, suggesting that the image captures a genuine moment of play."
Answer: real\n
Example 4:\n
Agent's response: "The image appears to be a real photograph of a person standing in an airport baggage claim area. The luggage carousel is visible, and there are suitcases and other travelers in the background. The person in the foreground is wearing a backpack and seems to be waiting for their luggage. The setting looks like a typical airport environment."
Answer: real\n
Example 5:\n
Agent's response: {}
Answer: """

def extract_answer(model, item):
    prediction = item['prediction']
    answer = item['answer']
    label = 1 if answer == 'fake' else 0
    if prediction.strip().lower() in ['fake', 'fake.']:
        extract = 'fake'
    elif prediction.strip().lower() in ['real', 'real.']:
        extract = 'real'
    else:
        prompt = EXTRACTION_FAKE_OR_REAL.format(prediction)
        extract = model.generate(prompt)
        extract = extract.strip().lower()
    if extract == 'fake':
        pred = 1
    elif extract == 'real':
        pred = 0
    else:
        pred = -1
    if extract == answer:
        return dict(hit=1, extract=extract, label=label, pred=pred)
    else:
        return dict(hit=0, extract=extract, label=label, pred=pred)


def eval_results(model, data, nproc, result_file):
    items = []
    result = {}
    if os.path.exists(result_file):
        result = load(result_file)

    for i in range(len(data)):
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(model=model, item=x) for x in items]
    keys = [x['index'] for x in items]

    if len(tups):
        res = track_progress_rich(extract_answer, tups, nproc=nproc, chunksize=nproc, save=None, keys=keys)
        for k,v in zip(keys, res):
            if k not in result:
                result[k] = v

    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['extract'] = [result[i]['extract'] for i in data['index']]
    data['label'] = [result[i]['label'] for i in data['index']]
    data['pred'] = [result[i]['pred'] for i in data['index']]
    return data


def report_metrics(data):
    label = list(data['label'])
    pred = list(data['pred'])

    label = [int(x) for x in label]
    pred = [int(x) for x in pred]
    label = np.array(label)
    pred = np.array(pred)

    # cal acc
    correct = (label == pred).sum().item()
    accuracy = correct / len(pred)

    # AP
    ap = metrics.average_precision_score(label, pred)

    fake_rate = (pred == 1.).sum().item() / len(pred)
    real_rate = (pred == 0.).sum().item() / len(pred)
    unknown_rate = (pred == -1.).sum().item() / len(pred)

    return {
        "accuracy": accuracy,
        "average precision": ap,
        "fake_rate": fake_rate,
        "real_rate": real_rate,
        "unknown_rate": unknown_rate
    }
