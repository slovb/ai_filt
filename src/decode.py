import email
import email.policy
from glob import glob
from pathlib import Path

import html2text
import re


class Logger:
    def __init__(self):
        self.num_processed = 0
        self.warnings = {}
        self.messages = {}

    def increment_processed(self) -> None:
        self.num_processed += 1

    def warning(self, name: str, message: str) -> None:
        print(f"Warning: {message}")
        if name not in self.warnings:
            self.warnings[name] = []
        self.warnings[name].append(message)

    def message(self, name: str, message: str) -> None:
        print(message)
        if name not in self.messages:
            self.messages[name] = []
        self.messages[name].append(message)

    def summary(self) -> str:
        num_warnings = len(self.warnings)
        output = []
        output.append(
            f"Summary: {self.num_processed} processed files, {num_warnings} warnings"
        )
        for filename in self.warnings:
            output.append(filename)
        return "\n".join(output)


class SpecificLogger:
    def __init__(self, logger: Logger, name: str):
        self.logger = logger
        self.name = name

    def warning(self, message: str) -> None:
        self.logger.warning(name=self.name, message=message)

    def message(self, message: str) -> None:
        self.logger.message(name=self.name, message=message)


def parse_html(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.ignore_images = True
    h.ignore_tables = True
    return h.handle(html)


def convert_to_lower(match_obj):
    if match_obj.group() is not None:
        return match_obj.group().lower()


def cleanup(content):
    # lowercase
    content = re.sub(r'[A-Z]', convert_to_lower, content)
    content = re.sub(r'Å', 'å', content)
    content = re.sub(r'Ä', 'ä', content)
    content = re.sub(r'Ö', 'ö', content)

    # anti email
    content = re.sub(r'([A-Za-z0-9åäö]+[.-_])*[A-Za-z0-9åäö]+@[A-Za-z0-9-åäö]+(\.[A-Z|a-z]{2,})+', '', content)

    # anti numbers
    content = re.sub(r'[0-9]', '', content)

    # anti symbols
    content = re.sub(r'[-,:\+\(\)]', '', content)
    content = re.sub(r'[\*\/]', ' ', content)
    # content = re.sub(r'\.', '\n', content) # beware dotnet

    # anti tab
    content = re.sub(r'\t', ' ', content)
    
    # only approved characters
    content = re.sub(r'[^a-zåäö \n]', '', content)
    
    # adjust whitespace
    content = re.sub(r'^[ \n]+', '', content)
    content = re.sub(r'  +', ' ', content)
    content = re.sub(r' *\n *', '\n', content)
    content = re.sub(r'\n\n+', '\n', content)

    return content


def convert_part(part, logger: SpecificLogger) -> str:
    if part.get_content_maintype() != "text":
        raise Exception("Unknown " + part.get_content_maintype())
    payload = part.get_payload(decode=True)
    charset = part.get_content_charset()
    if charset is None:
        charset = "utf-8"
        logger.message("No charset found, assuming UTF-8")
    if charset == "us-ascii":
        charset = "utf-8"
        logger.warning("Charset claimed us-ascii, assumed error so ignoring")
        return ""
    content = payload.decode(charset)
    if part.get_content_type() == "text/html":
        content = parse_html(content)
    elif part.get_content_type() != "text/plain":
        raise Exception("more decode cases needed " + part.get_content_type())
    return content


def convert_message(message, logger: SpecificLogger) -> str:
    if message.get_content_type() == "multipart/alternative":
        part = message.get_body(preferencelist=("html", "plain"))
        if part is None:
            raise Exception("more cases needed")
        content = convert_part(part=part, logger=logger)
    elif message.is_multipart():
        logger.warning(f"Weird multipart type: {message.get_content_type()}")
        contents = []
        for part in message.get_payload():
            if part.get_content_maintype() in ["application", "image"]:
                logger.warning(
                    f"Contains {part.get_content_maintype()} files, ignoring for now"
                )
                continue
            contents.append(convert_message(message=part, logger=logger))
        if len(contents) == 0:
            raise Exception("No understandable content found!!")
        content = "\n".join(contents)
    else:
        content = convert_part(part=message, logger=logger)
    return content


def read(filename: str, logger: Logger) -> str:
    """read a file and convert it to the text format"""
    with open(filename, "r") as f:
        message = email.message_from_file(f, policy=email.policy.default)
        # subject = message["subject"]
        content = convert_message(
            message=message, logger=SpecificLogger(logger=logger, name=filename)
        )
        return cleanup(content)
        # return f"{cleanup(subject)}\n\n{cleanup(content)}"


def convert(pathname: str, target: str, logger: Logger) -> None:
    """convert the elm files in pathname and place the converted .txt in target"""
    files = glob(pathname)
    for filename in files:
        print(f"processing: {filename}")
        outname = f"{target}/{Path(filename).stem}.txt"
        outdata = read(filename=filename, logger=logger)
        with open(outname, "w", encoding="utf8") as f:
            f.write(outdata)
        print(f"wrote: {outname}")
        logger.increment_processed()
        print(" ")


if __name__ == "__main__":
    logger = Logger()
    convert(
        pathname="data/raw/ej_intressanta/*.eml",
        target="data/decoded/ej_intressanta",
        logger=logger,
    )
    convert(
        pathname="data/raw/intressanta/*.eml",
        target="data/decoded/intressanta",
        logger=logger,
    )
    print(logger.summary())
