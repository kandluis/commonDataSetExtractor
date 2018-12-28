import os
import subprocess
import pandas as pd
import argparse
import enum
import editdistance
import unicodedata

import pprint

from typing import Any, Dict, Iterable, Text, List, Tuple, Callable, Optional

_FNULL = open(os.devnull, 'w')
_DATA_DIR = 'data'
_OUT_DIR = 'output'


class Section(enum.Enum):
  A = enum.auto()
  B = enum.auto()
  C = enum.auto()


def getPFDFilenames() -> Iterable[Tuple[Text, Text]]:
  '''Yields the list of PDF filenames.

  Yields:
    List of tuples of (dir, filename.pdf) to extract.
  '''
  for root, dirs, files in os.walk(_DATA_DIR):
    for file in files:
      if file.endswith('.pdf'):
        yield _DATA_DIR, file


def getTextContents() -> Iterable[List[Text]]:
  """Yields cleaned text contents"""
  for root, dirs, files in os.walk(os.path.join(_DATA_DIR, _OUT_DIR)):
    for file in files:
      if file.endswith('.txt'):
        with open(os.path.join(root, file)) as fp:
          yield [
              unicodedata.normalize("NFKD", line).strip()
              for line in fp.readlines()
          ]


def addToDict(key: Text,
              constructor=str) -> Callable[[Text, Dict[Text, Any]], None]:
  def fun(value: Text, acc: Dict[Text, Any]):
    val = value.strip()
    if val:
      acc[key] = constructor(val)

  return fun


# Map from "approximate matching line" to function to call to process it.
def retrieveProcessorForLine(
    description: Text) -> Optional[Callable[[Text, Dict[Text, Any]], None]]:
  kProcessors = [
      ('Name of College/University', addToDict('College Name')),
      ('First or only early decision plan closing date',
       addToDict('ED I Deadline')),
      ('First or only early decision plan notification date',
       addToDict('ED I Notification')),
      ('Other early decision plan closing date', addToDict('ED II Deadline')),
      ('Other early decision plan notification date',
       addToDict('ED II Notification')),
      ('Number of early decision applications received by your institution',
       addToDict('Number of ED Applicants', int)),
      ('Number of applicants admitted under early decision plan',
       addToDict('Number of ED Applicants Admitted')),
      ('Early action closing date', addToDict('EA Deadline')),
      ('Early action notification date', addToDict('EA Notification')),
      ('Total first­ time, first ­year (freshman) men who applied',
       addToDict('Men Applied')),
      ('Total first­ time, first­ year (freshman) women who applied',
       addToDict('Women Applied')),
      ('Total first­ time, first­ year (freshman) men who were admitted',
       addToDict('Men Admitted')),
      ('Total first­ time, first­ year (freshman) women who were admitted',
       addToDict('Women Admitted')),
      ('Total full­time, first­ time, first­ year (freshman) men who enrolled',
       addToDict('Men Enrolled')),
      ('Total full­time, first­ time, first­ year (freshman) women who enrolled',
       addToDict('Women Enrolled'))
  ]
  kThreshold = 0.1
  descLen = len(description)

  def ratio(text):
    distance = editdistance.eval(text, description)
    return distance / (len(text) + descLen)

  ratiosAndFunctions = sorted(
      [(ratio(trigger), func) for trigger, func in kProcessors],
      key=lambda x: x[0])
  minRatio, func = ratiosAndFunctions[0]
  if minRatio < kThreshold:
    return func
  return None


def isInterestingSection(section: Text) -> bool:
  kInterestingSections = {"A1", "C21"}
  return (len(section) == 2
          or len(section) == 3 and section in kInterestingSections)


def ExtractDataFromText(contents: List[Text]) -> Dict[Text, Any]:
  result = {}
  for line in contents:
    fragments = [fragment for fragment in line.split("  ") if fragment]
    if not fragments or len(fragments) <= 2: continue
    section, description, value = fragments[0], fragments[1], " ".join(
        fragments[2:])
    if not isInterestingSection(section): continue
    processor = retrieveProcessorForLine(description)
    if not processor: continue
    processor(value, result)

  return result


def ConvertToText() -> None:
  '''Converts input data into text.
  '''
  processes = []
  for root, filename in getPFDFilenames():
    infilepath = os.path.join(root, filename)
    basename = filename[:-len('.pdf')]
    outfilepath = os.path.join(root, _OUT_DIR, '%s.txt' % basename)
    command = ['gs', '-sDEVICE=txtwrite', '-o', outfilepath, infilepath]
    print("Running command %s." % " ".join(command))
    processes.append(subprocess.Popen(command, stdout=_FNULL))

  successful = 0
  for process in processes:
    if process.wait() == 0:
      successful += 1
    if successful % 10 == 0:
      print("Finished waiting for %s successful conversions." % successful)

  print("Finished %s successful conversion%s of %s." %
        (successful, "" if successful < 2 else "s", len(processes)))


def main():
  parser = argparse.ArgumentParser(description='Generate College Data Files')
  parser.add_argument(
      '--convert',
      type=bool,
      nargs=1,
      default=False,
      required=False,
      help=
      'Whether or not .pdf files in data/ should be dumped into data/output or if data/output should be used directly.'
  )
  args = parser.parse_args()
  if args.convert:
    ConvertToText()

  for textFileContents in getTextContents():
    data: Dict[Text, Any] = ExtractDataFromText(textFileContents)
    pprint.pprint(data)


if __name__ == '__main__':
  main()