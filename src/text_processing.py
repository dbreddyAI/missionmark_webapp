from bs4 import BeautifulSoup, Comment
import re


def strip_html(doc):

    if not doc:
        return ""

    soup = BeautifulSoup(doc, 'html.parser')
    pattern = re.compile(r"(?i)requirements")
    result = []

    for text_element in soup.find_all(text=True):

        if not _tag_visible(text_element):
            continue

        s = str(text_element).strip()
        if not s:
            continue

        if len(s) < 69:
            result.append(pattern.sub("REQUIREMENTS", s))
        else:
            result.append(s)


    return "\n".join(result)



def _tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'meta']:
        return False
    if isinstance(element, Comment):
        return False
    return True

