import cloudpickle
import hashlib
import io
import mlflow
import os
import requests

from PIL import Image
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_tools.core.utils.io import get_cache_dir, logger


class SquirrelDetectorLSModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(SquirrelDetectorLSModel, self).__init__(**kwargs)

        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        self.user = os.getenv("DAGSHUB_USER_NAME")
        self.token = os.getenv("DAGSHUB_TOKEN")
        self.repo = os.getenv("DAGSHUB_REPO_NAME")

        client = mlflow.MlflowClient()
        name = 'SquirrelDetector'
        version = client.get_latest_versions(name=name)[0].version

        self.model_version = f'{name}:{version}'

        model_uri = f'models:/{name}/{version}'

        self.model = mlflow.pyfunc.load_model(model_uri)

    def image_uri_to_https(self, uri):
        if uri.startswith('http'):
            return uri
        elif uri.startswith('repo://'):
            link_data = uri.split("repo://")[-1].split("/")
            commit, tree_path = link_data[0], "/".join(link_data[1:])
            return f"https://dagshub.com/api/v1/repos/{self.user}/{self.repo}/raw/{commit}/{tree_path}"
        raise FileNotFoundError(f'Unkown URI {uri}')

    def download_image(self, url):
        cache_dir = get_cache_dir()
        parsed_url = urlparse(url)
        url_filename = os.path.basename(parsed_url.path)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        filepath = os.path.join(cache_dir, url_hash + '__' + url_filename)
        if not os.path.exists(filepath):
            logger.info('Download {url} to {filepath}'.format(url=url, filepath=filepath))
            auth = HTTPBasicAuth(self.user, self.token)
            r = requests.get(url, stream=True, auth=auth)
            r.raise_for_status()
            with io.open(filepath, mode='wb') as fout:
                fout.write(r.content)
        return filepath

    def predict_task(self, task):
        uri = task['data']['image']
        url = self.image_uri_to_https(uri)
        image_path = self.download_image(url)

        img = Image.open(image_path)
        img_w, img_h = img.size

        objs = self.model.predict(img)

        lowest_conf = 2.0

        img_results = []

        for obj in objs:
            x, y, w, h, conf, cls = obj
            cls = int(cls)
            conf = float(conf)
            x = 100 * float(x - w / 2) / img_w
            y = 100 * float(y - h / 2) / img_h
            w = 100 * float(w) / img_w
            h = 100 * float(h) / img_h
            if conf < lowest_conf:
                lowest_conf = conf
            label = self.labels[cls]

            img_results.append({
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [label],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                },
                'score': conf
            })

        result = {
            'result': img_results,
            'model_version': self.model_version,
            'task': task['id']
        }

        if lowest_conf <= 1.0:
            result['score'] = lowest_conf

        url = f'https://dagshub.com/{self.user}/{self.repo}/annotations/git/api/predictions/'
        auth = HTTPBasicAuth(self.user, self.token)
        res = requests.post(url, auth=auth, json=result)
        if res.status_code != 200:
            print(res)


    def predict(self, tasks, **kwargs):
        """ This is where inference happens: 

            from PIL import Image
            model returns the list of predictions based on input list of tasks
            
            :param tasks: Label Studio tasks in JSON format
        """
        for task in tasks:
            self.predict_task(task)

        return []

    def fit(self, completions, workdir=None, **kwargs):
        """ This is where training happens: train your model given list of completions,
            then returns dict with created links and resources

            :param completions: aka annotations, the labeling results from Label Studio 
            :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result
        return {'random': random.randint(1, 10)}
