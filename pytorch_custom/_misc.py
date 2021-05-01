class text_styles:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


class TrivialContext:
    def __enter__(self):
        pass
    def __exit__(*args):
        pass