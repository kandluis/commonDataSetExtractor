import os
import subprocess
import pandas as pd
import argparse
import enum
import editdistance
import unicodedata
import datetime

import pprint

from typing import Any, Dict, Iterable, List, Tuple, Callable, Optional

_FNULL = open(os.devnull, 'w')
_DATA_DIR = 'data'
_OUT_DIR = 'output'


class Section(enum.Enum):
  A = enum.auto()
  B = enum.auto()
  C = enum.auto()


def getPFDFilenames() -> Iterable[Tuple[str, str]]:
  '''Yields the list of PDF filenames.

  Yields:
    List of tuples of (dir, filename.pdf) to extract.
  '''
  for root, dirs, files in os.walk(_DATA_DIR):
    for file in files:
      if file.endswith('.pdf'):
        yield _DATA_DIR, file


def getTextContents() -> Iterable[List[str]]:
  """Yields cleaned str contents"""
  for root, dirs, files in os.walk(os.path.join(_DATA_DIR, _OUT_DIR)):
    for file in files:
      if file.endswith('.txt'):
        with open(os.path.join(root, file)) as fp:
          yield [
              unicodedata.normalize("NFKD", line).strip()
              for line in fp.readlines()
          ]


def addToDict(key: str,
              constructor=str) -> Callable[[str, Dict[str, Any]], None]:
  def fun(value: str, acc: Dict[str, Any]):
    val = value.strip()
    if val:
      acc[key] = constructor(val)

  return fun


def percent(x: str) -> float:
  return float(x.strip('%')) / 100


_CURRENT_YEAR = datetime.datetime.now().year


def date(x: str) -> datetime.datetime:
  date = datetime.datetime.strptime(x, '%m/%d')
  if date.month < 6:
    return date.replace(year=_CURRENT_YEAR + 1)
  return date.replace(year=_CURRENT_YEAR)


# Map from "approximate matching line" to function to call to process it.
def retrieveProcessorForLine(
    description: str) -> Optional[Callable[[str, Dict[str, Any]], None]]:
  kProcessors = [
      ('Name of College/University', addToDict('College Name'), None),
      ('First or only early decision plan closing date',
       addToDict('ED I Deadline', date), None),
      ('First or only early decision plan notification date',
       addToDict('ED I Notification', date), None),
      ('Other early decision plan closing date',
       addToDict('ED II Deadline', date), None),
      ('Other early decision plan notification date',
       addToDict('ED II Notification', date), None),
      ('Number of early decision applications received by your institution',
       addToDict('Number of ED Applicants', int), None),
      ('Number of applicants admitted under early decision plan',
       addToDict('Number of ED Applicants Admitted', int), None),
      ('Early action closing date', addToDict('EA Deadline', date), None),
      ('Early action notification date', addToDict('EA Notification', date),
       None),
      ('Total first­-time, first ­year (freshman) men who applied',
       addToDict('Men Applied', int), None),
      ('Total first­-time, first-year (freshman) women who applied',
       addToDict('Women Applied', int), None),
      ('Total first­-time, first-year (freshman) men who were admitted',
       addToDict('Men Admitted', int), None),
      ('Total first­-time, first-year (freshman) women who were admitted',
       addToDict('Women Admitted', int), None),
      # Special threhold needed for these two since there are other very
      # similar values in the PDF (for part-time) students
      ('Total full-time, first­-time, first-­year (freshman) men who enrolled',
       addToDict('Men Enrolled', int), 0.01),
      ('Total full-time, first­-time, first­-year (freshman) women who enrolled',
       addToDict('Women Enrolled', int), 0.01),
      ('Percent who had GPA of 3.75 and higher', addToDict(
          'GPA >3.75', percent), None),
      ('Percent who had GPA between 3.50 and 3.74',
       addToDict('GPA >3.50', percent), None),
      ('Percent who had GPA between 3.25 and 3.49',
       addToDict('GPA >3.25', percent), None),
      ('Percent who had GPA between 3.00 and 3.24',
       addToDict('GPA >3.00', percent), None),
      ('Percent who had GPA between 3.00 and 3.24',
       addToDict('GPA >2.75', percent), None),
      ('Percent who had GPA between 2.50 and 2.99',
       addToDict('GPA >2.50', percent), None),
      ('Percent who had GPA between 2.00 and 2.49',
       addToDict('GPA >2.00', percent), None),
      ('Percent who had GPA between 1.00 and 2.99',
       addToDict('GPA >1.00 ', percent), None)
  ]
  kThreshold = 0.075
  descLen = len(description)
  ratiosAndFunctions = []
  for trigger, func, threshold in kProcessors:
    distance = editdistance.eval(trigger, description)
    ratio = distance / (len(trigger) + descLen)
    if ((not threshold and ratio < kThreshold)
        or (threshold and ratio < threshold)):
      ratiosAndFunctions.append((ratio, func))
  if not ratiosAndFunctions:
    return None
  _, func = sorted(ratiosAndFunctions, key=lambda x: x[0])[0]
  return func


def isInterestingSection(section: str) -> bool:
  kInterestingSections = {"A1", "C11", "C21"}
  return (len(section) == 2
          or len(section) == 3 and section in kInterestingSections)


def ExtractDataFromText(contents: List[str]) -> Dict[str, Any]:
  result: Dict[str, Any] = {}
  iterator = iter(contents)
  for line in iterator:
    fragments = [fragment for fragment in line.split("  ") if fragment]
    if not fragments or len(fragments) <= 2: continue
    section, description, value = fragments[0], fragments[1], " ".join(
        fragments[2:])
    if not isInterestingSection(section): continue
    processor = retrieveProcessorForLine(description)
    if not processor: continue
    print(description)
    processor(value, result)

  return result


def ConvertTostr() -> None:
  '''Converts input data into str.
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
    ConvertTostr()

  for textFileContents in getTextContents():
    data: Dict[str, Any] = ExtractDataFromText(textFileContents)
    pprint.pprint(data)


if __name__ == '__main__':
  main()