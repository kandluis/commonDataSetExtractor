import os
import subprocess
import pandas as pd
import argparse
import enum
import editdistance
import unicodedata
import datetime

import pprint

from typing import Any, Dict, Iterable, List, Tuple, Callable, Optional, Set, Union

_FNULL = open(os.devnull, 'w')
_DATA_DIR: str = 'data'
_OUT_DIR: str = 'output'
_CURRENT_YEAR = datetime.datetime.now().year

_DEBUG: Optional[bool] = None
_PRINT_FAILURES: Optional[bool] = None


def getPFDFilenames() -> Iterable[Tuple[str, str]]:
  '''Yields the list of PDF filenames.

  Yields:
    List of tuples of (dir, filename.pdf) to extract.
  '''
  for root, dirs, files in os.walk(_DATA_DIR):
    for file in files:
      if file.endswith('.pdf'):
        yield _DATA_DIR, file


def getTextContentsAndName(
    match: str = None) -> Iterable[Tuple[str, List[str]]]:
  """Yields cleaned str contents"""
  for root, dirs, files in os.walk(os.path.join(_DATA_DIR, _OUT_DIR)):
    for file in files:
      if (file.endswith('.txt')
          and (not match or match.lower() in file.lower())):
        with open(os.path.join(root, file)) as fp:
          yield (file, [
              unicodedata.normalize("NFKD", line).strip()
              for line in fp.readlines()
          ])


def percent(x: str) -> Union[float, str]:
  try:
    return float(x.strip('%')) / 100
  except ValueError:
    return "N/A"


def date(x: str) -> Union[datetime.datetime, str]:
  if 'Mid-December'.lower() in x.lower():
    return 'Mid-December'
  kCandidateFormats = ['%m/%d', '%d-%b', '%b %d', '%b. %d']
  tried = 0
  while True:
    try:
      date = datetime.datetime.strptime(x, kCandidateFormats[tried])
      break
    except ValueError as e:
      tried += 1
      if tried >= len(kCandidateFormats):
        print("Error: %s" % e)
        return "N/A"
  if date.month < 6:
    return date.replace(year=_CURRENT_YEAR + 1)
  return date.replace(year=_CURRENT_YEAR)


def genericInt(x: str) -> Union[int, str]:
  try:
    return int(x.replace(',', ''))
  except:
    return "N/A"


def genericFloat(x: str) -> Union[float, str]:
  try:
    return float(x.replace(',', ''))
  except:
    return "N/A"


class ProcessorResult(enum.Enum):
  NEXT_LINE = enum.auto()
  SUCCESS = enum.auto()
  FAILED = enum.auto()


def addToDict(key: str, constructor=str
              ) -> Callable[[str, Dict[str, Any]], ProcessorResult]:
  def fun(value: str, acc: Dict[str, Any]) -> ProcessorResult:
    if not value: return ProcessorResult.NEXT_LINE
    val = value.strip()
    if val:
      acc[key] = constructor(val)
      return ProcessorResult.SUCCESS
    return ProcessorResult.NEXT_LINE

  return fun


def processSATPercentiles(
    key: str) -> Callable[[str, Dict[str, Any]], ProcessorResult]:
  def fun(value: str, acc: Dict[str, Any]) -> ProcessorResult:
    fragments = [val for val in value.split(" ") if val]
    if not fragments: return ProcessorResult.NEXT_LINE
    if len(fragments) > 1:
      acc["SAT Math (%s)" % key] = percent(fragments[-1])
      acc["SAT Reading (%s)" % key] = percent(fragments[-2])
      return ProcessorResult.SUCCESS
    return ProcessorResult.FAILED

  return fun


def processAverageGPA(value: str, acc: Dict[str, Any]) -> ProcessorResult:
  if not value:
    # We want to grab the next line since the snippet above matched pretty well.
    return ProcessorResult.NEXT_LINE
  fragments = [val for val in value.split(":") if val]
  if not fragments or len(fragments) < 2:
    return ProcessorResult.FAILED
  acc['Average GPA'] = genericFloat(fragments[-1])
  return ProcessorResult.SUCCESS


def processSATReadingPercentiles(value: str,
                                 acc: Dict[str, Any]) -> ProcessorResult:
  if not value:
    return ProcessorResult.NEXT_LINE
  fragments = [val for val in value.split(" ") if val]
  if len(fragments) >= 2:
    acc["SAT Reading 75%"] = genericInt(fragments[-1])
    acc["SAT Reading 25%"] = genericInt(fragments[-2])
    return ProcessorResult.SUCCESS
  return ProcessorResult.FAILED


def processSATMathPercentiles(value: str,
                              acc: Dict[str, Any]) -> ProcessorResult:
  fragments = [val for val in value.split(" ") if val]
  if len(fragments) >= 2:
    acc["SAT Math 75%"] = genericInt(fragments[-1])
    acc["SAT Math 25%"] = genericInt(fragments[-2])
    return ProcessorResult.SUCCESS
  return ProcessorResult.FAILED


_PROCESSORS = [
    ('Name of College/University', addToDict('College Name'), None),
    ('First or only early decision plan closing date',
     addToDict('ED I Deadline', date), None),
    ('First or only early decision plan notification date',
     addToDict('ED I Notification', date), None),
    ('Other early decision plan closing date', addToDict(
        'ED II Deadline', date), None),
    ('Other early decision plan notification date',
     addToDict('ED II Notification', date), None),
    ('Number of early decision applications received by your institution',
     addToDict('Number of ED Applicants', genericInt), None),
    ('Number of applicants admitted under early decision plan',
     addToDict('Number of ED Applicants Admitted', genericInt), None),
    ('Early action closing date', addToDict('EA Deadline', date), None),
    ('Early action notification date', addToDict('EA Notification', date),
     None),
    ('Total first­-time, first-year (freshman) men who applied',
     addToDict('Men Applied', genericInt), None),
    ('Total first­-time, first-year (freshman) women who applied',
     addToDict('Women Applied', genericInt), None),
    ('Total first­-time, first-year (freshman) men who were admitted',
     addToDict('Men Admitted', genericInt), None),
    ('Total first­-time, first-year (freshman) women who were admitted',
     addToDict('Women Admitted', genericInt), None),
    # Special threhold needed for these two since there are other very
    # similar values in the PDF (for part-time) students
    ('Total full-time, first­-time, first-­year (freshman) men who enrolled',
     addToDict('Men Enrolled', genericInt), 0.04),
    ('Total full-time, first­-time, first­-year (freshman) women who enrolled',
     addToDict('Women Enrolled', genericInt), 0.04),
    ('Percent who had GPA of 3.75 and higher', addToDict('GPA >3.75', percent),
     None),
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
     addToDict('GPA >1.00 ', percent), None),
    ('Average high school GPA of all degree­-seeking, first-­time, first­-year',
     processAverageGPA, None),
    ('SAT Evidence­-Based Reading', processSATReadingPercentiles, 0.2),
    ('SAT Critical Reading', processSATReadingPercentiles, 0.2),
    ('SAT Evidence­-', processSATReadingPercentiles, 0.2),
    ('SAT Evidence­-Based Reading and Writing', processSATReadingPercentiles,
     0.2),
    ('SAT Math', processSATMathPercentiles, 0.2),
    ('700­-800', processSATPercentiles('700-800'), 0.15),
    ('600-699', processSATPercentiles('600-699'), 0.15),
    ('500-599', processSATPercentiles('500-599'), 0.15),
    ('400-499', processSATPercentiles('400-499'), 0.15),
    ('300-399', processSATPercentiles('300-399'), 0.15),
    ('200-299', processSATPercentiles('200-299'), 0.15),
]

# List of sets of triggers which are mutually exclusive.
_SAME_PROCESSORS: List[Set[str]] = [{
    'SAT Evidence­-Based Reading', 'SAT Critical Reading', 'SAT Evidence­-',
    'SAT Evidence­-Based Reading and Writing'
}]


# Map from "approximate matching line" to function to call to process it.
def retrieveProcessorForLine(
    description: str, processed: Set[str]
) -> Optional[Tuple[str, Callable[[str, Dict[str, Any]], ProcessorResult]]]:
  kThreshold = 0.075
  descLen = len(description)
  ratiosAndFunctions = []
  for trigger, func, threshold in _PROCESSORS:
    if trigger in processed: continue
    distance = editdistance.eval(trigger, description)
    ratio = distance / (len(trigger) + descLen)
    if _DEBUG:
      print(description, trigger, ratio)
    if ((not threshold and ratio < kThreshold)
        or (threshold and ratio < threshold)):
      ratiosAndFunctions.append((ratio, trigger, func))
  if not ratiosAndFunctions:
    return None
  _, trigger, func = sorted(ratiosAndFunctions, key=lambda x: x[0])[0]
  if _DEBUG:
    print()
    print("Using trigger: %s for line: %s" % (trigger, description))
    print()
  return trigger, func


def isInterestingSection(section: str) -> bool:
  kInterestingSections = {"A1", "C9", "C1", "C11", "C12", "C21", "C22"}
  return ((len(section) == 2 or len(section) == 3)
          and section.strip() in kInterestingSections)


def ExtractDataFromText(filename: str, contents: List[str]) -> Dict[str, Any]:
  collegeInfo: Dict[str, Any] = {'filename': filename}
  processed: Set[str] = set()
  i: int = 0
  while i < len(contents):
    line = contents[i]
    fragments = [fragment.strip() for fragment in line.split(" ") if fragment]
    if not fragments:
      i += 1
      continue
    section = fragments[0]
    if not isInterestingSection(section):
      i += 1
      continue
    fragments = [
        fragment.strip() for fragment in line[len(section) + 1:].split("  ")
        if fragment
    ]
    if not fragments:
      i += 1
      line = contents[i]
      fragments = [
          fragment.strip() for fragment in line.split(" ") if fragment
      ]
      if not fragments: continue
      section = fragments[0]
      if isInterestingSection(section): continue
      fragments = [
          fragment.strip() for fragment in line.split("  ") if fragment
      ]
    description, value = fragments[0], " ".join(fragments[1:])
    if _DEBUG:
      print()
      print(section)
      print()
    triggerAndProcessor = retrieveProcessorForLine(description, processed)
    if not triggerAndProcessor:
      i += 1
      continue
    trigger, processor = triggerAndProcessor
    if _DEBUG:
      print()
      print("Processor is given value: %s" % value)
      print()
    result = processor(value, collegeInfo)
    while result == ProcessorResult.NEXT_LINE and i < len(contents):
      i += 1
      line = contents[i]
      fragments = [fragment for fragment in line.split(" ") if fragment]
      if not fragments:
        i += 1
        continue
      section = fragments[0]
      # We skip this processor and rewind (outer loop increments)
      if isInterestingSection(section):
        i -= 1
        break
      value = " ".join(
          [value] +
          [fragment.strip() for fragment in line.split("  ") if fragment])
      if _DEBUG:
        print("Processor is given value: %s" % value)
      result = processor(value, collegeInfo)

    if result == ProcessorResult.SUCCESS:
      processed.add(trigger)
      for sameSets in _SAME_PROCESSORS:
        if trigger in sameSets:
          for other in sameSets:
            processed.add(other)
    i += 1

  if _DEBUG or _PRINT_FAILURES:
    print()
    print()
    triggers: Set[str] = {trigger for trigger, _, _ in _PROCESSORS}
    failed: Set[str] = triggers - processed
    for failure in sorted(failed):
      print("Failed to extract information for %s" % failure)
  return collegeInfo


def ConvertToText() -> None:
  '''Converts input data into str.
  '''
  processes = []
  for root, filename in getPFDFilenames():
    infilepath = os.path.join(root, filename)
    basename = filename[:-len('.pdf')]
    outfilepath = os.path.join(root, _OUT_DIR, '%s.txt' % basename)
    command = ['gs', '-sDEVICE=txtwrite', '-o', outfilepath, infilepath]
    if _DEBUG: print("Running command %s." % " ".join(command))
    processes.append(subprocess.Popen(command, stdout=_FNULL))

  successful = 0
  for process in processes:
    if process.wait() == 0:
      successful += 1
    if successful % 10 == 0:
      print("Finished waiting for %s successful conversions." % successful)

  print("Finished %s successful conversion%s of %s." %
        (successful, "" if successful < 2 else "s", len(processes)))


def add_bool_arg(parser: argparse.ArgumentParser,
                 name: str,
                 default: bool = False) -> None:
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--' + name, dest=name, action='store_true')
  group.add_argument('--no-' + name, dest=name, action='store_false')
  parser.set_defaults(**{name: default})


def getTable(collegeFilter: str) -> pd.DataFrame:
  data = [
      ExtractDataFromText(name, textFileContents)
      for name, textFileContents in getTextContentsAndName(collegeFilter)
  ]
  return pd.DataFrame.from_dict(data)


def main():
  parser = argparse.ArgumentParser(description='Generate College Data Files')
  parser.add_argument('--college', default=None, type=str)
  add_bool_arg(parser, 'convert')
  add_bool_arg(parser, 'debug')
  add_bool_arg(parser, 'print_failures')
  args = parser.parse_args()
  if args.convert:
    ConvertToText()
  global _DEBUG, _PRINT_FAILURES
  _DEBUG = args.debug
  _PRINT_FAILURES = args.print_failures

  getTable(args.college).to_csv('data/results.csv')


if __name__ == '__main__':
  main()