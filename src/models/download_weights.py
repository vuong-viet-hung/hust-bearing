import gdown


WEIGHTS_URL = 'https://drive.google.com/drive/folders/1zIq2a6CX4L_V_ERTmA6gSTYsuGZj8wvi?usp=sharing'


if __name__ == '__main__':
    gdown.download_folder(WEIGHTS_URL)
