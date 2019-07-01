"""Control global constant variable with Config class"""
import json


class Config(dict):
    """JSON loadable config"""

    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        self._parse_dict()

    @classmethod
    def load_from_json(cls, file_path: str) -> "Config":
        """load and parse to Config instance from JSON file path

        :param file_path: JSON file location
        :return: Initiated Config Class
        """
        with open(file_path) as file:
            return cls(json.load(file))

    def _parse_dict(self):
        """parse dictionary as hierarchical Config class"""
        for key, value in self.items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        """forward attribute -> key reference for assign """
        if isinstance(value, dict):
            raise TypeError("dict type config value should be Config class instance")
        self[key] = value

    def __getattr__(self, name):
        """forward attribute -> key reference"""
        if name not in self:
            raise AttributeError("No such attribute: " + name)
        return self[name]

    def __delattr__(self, name):
        """forward attribute -> key reference for delete"""
        if name not in self:
            raise AttributeError("No such attribute: " + name)
        del self[name]
