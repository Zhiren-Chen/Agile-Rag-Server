class bcolors:

    OK = "\033[92m"  
    WARNING = "\033[93m"  
    FAIL = "\033[91m"  
    RESET = "\033[0m" 
    BLUE = "\033[96m"
    PURPLE = "\033[95m"

class Disp(bcolors):

    def white(self, text, *args):
        print(text, *args)

    def green(self, text, *args):
        print(self.OK + text + ' ' + ' '.join(args) + self.RESET)
    
    def yellow(self, text, *args):
        print(self.WARNING + text + ' ' + ' '.join(args) + self.RESET)

    def red(self, text, *args):
        print(self.FAIL + text + ' ' + ' '.join(args) + self.RESET)

    def blue(self, text, *args):
        print(self.BLUE + text + ' ' + ' '.join(args) + self.RESET)

    def purple(self, text, *args):
        print(self.PURPLE + text + ' ' + ' '.join(args) + self.RESET)

disp = Disp()

if __name__ == "__main__":
    disp = Disp()
    disp.white("hello,", "this is a message.")
    disp.green("hello,", "this is a success.")
    disp.yellow("hello,", "this is a warning.")
    disp.red("hello,", "this is an error.")
    disp.blue("hello,", "this is something interesting.")
    disp.purple("hello,", "this is something fancy.")
