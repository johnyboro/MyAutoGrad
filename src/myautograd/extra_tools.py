import subprocess


def print_to_kitty(dot) -> None:
    png_data = dot.pipe()
    process = subprocess.Popen(['kitten', 'icat'], stdin=subprocess.PIPE)
    process.communicate(input=png_data)

