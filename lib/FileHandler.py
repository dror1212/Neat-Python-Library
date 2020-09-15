import pickle

class FileHandler:
    @staticmethod
    def save(data, fileName):
        with open(fileName, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def read(fileName):
        data = None
        try:
            with open(fileName, 'rb') as file:
                data = pickle.load(file)
        except:
            with open(fileName, 'wb') as file:
                pass
        return data
