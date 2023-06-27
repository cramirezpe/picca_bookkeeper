
from typing import *
import collections 
import copy

class DictUtils:
    @staticmethod
    def merge_dicts(dict1: Dict, dict2: Dict):
        """Merges two dictionaries recursively preserving values in dict2"""
        result = copy.deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.abc.Mapping):
                if not isinstance(result.get(key, {}), collections.abc.Mapping):
                    result[key] = dict()
                result[key] = DictUtils.merge_dicts(result.get(key, {}), value)
            else:
                result[key] = copy.deepcopy(dict2[key])

        return result


    @staticmethod
    def remove_matching(dict1: Dict, dict2: Dict):
        """Removes occurrences happening in two dictionaries."""
        result = copy.deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.abc.Mapping):
                result[key] = DictUtils.remove_matching(result.get(key, {}), value)
            elif key in result:
                result.pop(key)

        return result

    @staticmethod
    def diff_dicts(dict1: Dict, dict2: Dict):
        """Function to give differences between dicts"""
        differences = dict()

        for key, value in dict1.items():
            if key not in dict2:
                differences[key] = [dict1[key], None]
            elif isinstance(value, collections.abc.Mapping):
                differences[key] = DictUtils.diff_dicts(dict1[key], dict2[key])
            elif not dict1[key] == dict2[key]:
                differences[key] = [dict1[key], dict2[key]]
            
        for key, value in dict2.items():
            if key not in dict1:
                differences[key] = [None, dict1[key]]

        return differences

    @staticmethod
    def remove_empty(config: Dict):
        """Function to remove empty strings from dict"""
        result = copy.deepcopy(config)
        
        while True: 
            for key, value in result.items():
                if value == "" or value == {}:
                    result.pop(key)
                    break
                elif isinstance(value, collections.abc.Mapping):
                    result[key] = DictUtils.remove_empty(value)
            else:
                # This trick allows for repeating the function until
                # input and outut are the same: all empty strings have
                # been removed.
                # This happens because the function in one iteration may
                # remove all items in one dict, and then we need an
                # extra iteration to remove the dict itself.
                if result != config:
                    result = DictUtils.remove_empty(result)
                return result
