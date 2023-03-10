class PathNotDirectoryException(Exception):
    
    def __init__(self, path: str, message="The path provided is not a directory: ") -> None:
        self.message = message + path
        super().__init__(self.message)