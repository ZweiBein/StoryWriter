def remove_non_ascii(s):
    return "".join(i for i in s if 97 <= ord(i) <= 128 or ord(i) == 32 or i == '.' or i == '!' or i == '?')

