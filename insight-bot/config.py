import yaml

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def zoom_meeting_url(self):
        return self.get('zoom_meeting_url')

    @property
    def output_dir(self):
        return self.get('output_dir', '/tmp')

    @property
    def sample_rate(self):
        return self.get('sample_rate', 16000)

    @property
    def channels(self):
        return self.get('channels', 1)
