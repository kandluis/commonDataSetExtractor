import os
import subprocess
import pandas as pd
import argparse
import enum
import editdistance
import unicodedata
import datetime
import ssl

from urllib import request
from concurrent import futures

import pprint

from typing import Any, Dict, Iterable, List, Tuple, Callable, Optional, Set, Union

_FNULL = open(os.devnull, 'wb')
_DATA_DIR: str = 'data'
_OUT_DIR: str = 'output'
_CURRENT_YEAR = datetime.datetime.now().year

_DEBUG: Optional[bool] = None
_PRINT_FAILURES: Optional[bool] = None


def getPDFFilenames() -> Iterable[Tuple[str, str, str]]:
  '''Yields the list of PDF filenames.

  Yields:
    List of tuples of (dir, filename.pdf, processor) to extract.
  '''
  data = pd.read_csv(os.path.join(_DATA_DIR, "links.csv"), header=0)
  processors = {
      "%s.pdf" % GetFileName(row.School): row.Processor
      for row in data.itertuples()
  }
  data.School = data.School.apply(lambda x: GetFileName(x))
  for root, dirs, files in os.walk(_DATA_DIR):
    for file in files:
      if file.endswith('.pdf'):
        yield _DATA_DIR, file, processors[file.lower()]


def getTextContentsAndName(
    match: str = None) -> Iterable[Tuple[str, List[str]]]:
  """Yields cleaned str contents"""
  for root, dirs, files in os.walk(os.path.join(_DATA_DIR, _OUT_DIR)):
    for file in files:
      if (file.endswith('.txt')
          and (not match or match.lower() in file.lower())):
        with open(os.path.join(root, file), 'r+', encoding="ISO-8859-1") as fp:
          yield (file, [
              unicodedata.normalize("NFKD", line).strip()
              for line in fp.readlines()
          ])


def percent(x: str) -> Optional[float]:
  try:
    return float(x.strip('%')) / 100
  except ValueError:
    return None


def date(x: str) -> Union[datetime.datetime, str]:
  kTextMatches = [
      'Mid-December', 'Mid December', 'Mid-February', 'Mid February',
      'Rolling', 'late January'
  ]
  for match in kTextMatches:
    if match.lower() in x.lower():
      return match

  def generateDatesMatches(dateformat: str) -> List[str]:
    return [
        "%s%s" % (dateformat, suffix)
        for suffix in ("", "st", "nd", "rd", "th")
    ]

  kCandidateFormats = [
      match for dateformat in
      ['%m/%d', '%d-%b', '%b %d', '%b. %d', '%m/%d/%Y', "%B %d", "-%d-%b"]
      for match in generateDatesMatches(dateformat)
  ]
  tried = 0
  while True:
    try:
      date = datetime.datetime.strptime(x, kCandidateFormats[tried])
      break
    except ValueError as e:
      tried += 1
      if tried >= len(kCandidateFormats):
        if _DEBUG: print("Error: %s" % e)
        return x
  if date.month < 6:
    return date.replace(year=_CURRENT_YEAR + 1)
  return date.replace(year=_CURRENT_YEAR)


def genericInt(x: str) -> Optional[int]:
  try:
    return int(x.replace(',', ''))
  except:
    return None


def genericFloat(x: str) -> Optional[float]:
  try:
    return float(x.replace(',', ''))
  except:
    return None


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
    section = fragments[0].replace('.', '')
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
      section = fragments[0].replace('.', '')
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
      section = fragments[0].replace('.', '')
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

  triggers: Set[str] = {trigger for trigger, _, _ in _PROCESSORS}
  failed: Set[str] = triggers - processed
  # Always print those with a high failure rate.
  if _DEBUG or _PRINT_FAILURES:
    print()
    print()
    for failure in sorted(failed):
      print(
          "Failed to extract information for %s from %s" % (failure, filename))
  if len(failed) / len(triggers) > 0.5:
    print("Failed to Process %s" % filename)

  return collegeInfo


def GetFileName(schoolName: str) -> str:
  return schoolName.replace(' ', '_').lower()


def DownloadPdfs(collegeFilter: Optional[str]) -> None:
  '''Downloads PDFs for the schools as specified in data/links.csv'''

  context = ssl._create_unverified_context()
  request_headers = {
      'User-Agent': 'curl/7.47.1',
      'content-type': 'application/pdf'
  }

  def download_file(info: Tuple[str, str]) -> None:
    filepath: str = info[0]
    download_url: str = info[1]
    try:
      query = request.Request(download_url, headers=request_headers)
      response = request.urlopen(query, context=context)
      with open(filepath, 'wb') as file:
        file.write(response.read())
    except Exception as e:
      print("Failed on URL: %s" % download_url)
      if _DEBUG: print(e)

  data: pd.DataFrame = pd.read_csv(
      os.path.join(_DATA_DIR, 'links.csv'), header=0)
  dropped = data[data.Download.isnull()]
  data = data[~data.Download.isnull()]
  arguments = [
      (os.path.join(_DATA_DIR, "%s.pdf" % (GetFileName(row.School))),
       row.Download) for row in data.itertuples() if row.Download and (
           not collegeFilter or collegeFilter.lower() in row.School.lower())
  ]
  print("Dropped %s values without URLs." % (len(dropped)))
  if _DEBUG or _PRINT_FAILURES:
    for school, url in zip(dropped.School, dropped['General Link']):
      print("Did not downloand PDF for %s. Missing URL. General URL is: %s" %
            (school, url))
  with futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(download_file, arguments)

  # This waits for all to be done.
  for result in results:
    pass


def ConvertToText(collegeFilter: Optional[str]) -> None:
  '''Converts input data into str.
  '''
  processes: List[Tuple[str, Any]] = []
  for root, filename, processor in getPDFFilenames():
    infilepath = os.path.join(root, filename)
    basename = filename[:-len('.pdf')]
    if (collegeFilter is None or editdistance.eval(collegeFilter, basename) /
        (len(collegeFilter) + len(basename)) < 0.5):
      outfilepath = os.path.join(root, _OUT_DIR, '%s.txt' % basename)
      command = None
      if processor.lower() == "gs":
        command = ['gs', '-sDEVICE=txtwrite', '-o', outfilepath, infilepath]
      elif processor.lower() == "./pdftotext":
        command = [
            './pdftotext', '-layout', '-eol', 'unix', '-nopgbrk', infilepath,
            outfilepath
        ]
      else:
        raise Exception("Processor not implemeneted: %s" % processor)

      if _DEBUG:
        print("Running command %s" % " ".join(command))
      processes.append((filename, subprocess.Popen(command, stdout=_FNULL)))

  successful = 0
  failed: List[str] = []
  for filename, process in processes:
    if process.wait() == 0:
      successful += 1
    else:
      failed.append(filename)
    if successful % 10 == 0:
      if _DEBUG:
        print("Finished waiting for %s successful conversions." % successful)

  print("Finished %s successful conversion%s of %s." %
        (successful, "" if successful < 2 else "s", len(processes)))
  if _DEBUG or _PRINT_FAILURES:
    for failure in failed:
      print("Failed to convert file: %s" % failure)


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


def analyzeRawData(data: pd.DataFrame) -> pd.DataFrame:
  totalEnrolled, totalAdmitted, totalApplicants = None, None, None
  if 'Women Enrolled' in data.columns and 'Men Enrolled' in data.columns:
    totalEnrolled = data["Women Enrolled"].add(
        data["Men Enrolled"], fill_value=0)
  if 'Women Applied' in data.columns and 'Men Applied' in data.columns:
    totalApplicants = data["Women Applied"].add(
        data["Men Applied"], fill_value=0)
  if 'Women Admitted' in data.columns and 'Men Admitted' in data.columns:
    totalAdmitted = data["Women Admitted"].add(
        data["Men Admitted"], fill_value=0)
  if totalAdmitted is not None and totalApplicants is not None:
    data["Admissions Rate"] = totalAdmitted / totalApplicants
  if totalEnrolled is not None and totalAdmitted is not None:
    data["Yield"] = totalEnrolled / totalAdmitted
  if 'Men Admitted' in data.columns and 'Men Applied' in data.columns:
    data["Men Admissions Rate"] = data["Men Admitted"] / data["Men Applied"]
  if 'Women Admitted' in data.columns and 'Women Applied' in data.columns:
    data["Women Admissions Rate"] = data["Women Admitted"] / data[
        "Women Applied"]
  if 'Number of ED Applicants' in data.columns and 'Number of ED Applicants Admitted' in data.columns:
    data['ED Admissions Rate'] = data['Number of ED Applicants Admitted'].div(
        data['Number of ED Applicants'], fill_value=0)
  return data


def main():
  parser = argparse.ArgumentParser(description='Generate College Data Files')
  parser.add_argument('--college', default=None, type=str)
  add_bool_arg(parser, 'convert')
  add_bool_arg(parser, 'download')
  add_bool_arg(parser, 'debug')
  add_bool_arg(parser, 'print_failures')
  args = parser.parse_args()
  global _DEBUG, _PRINT_FAILURES
  _DEBUG = args.debug
  _PRINT_FAILURES = args.print_failures

  if args.download:
    DownloadPdfs(args.college)
  if args.convert:
    ConvertToText(args.college)

  results = getTable(args.college)
  results = analyzeRawData(results)

  OUTPUT_COLUMNS = [
      column for column in (
          'College Name', 'Admissions Rate', 'Yield', 'Men Admissions Rate',
          'Women Admissions Rate', 'Men Applied', 'Men Admitted',
          'Men Enrolled', 'Women Admitted', 'Women Applied', 'Women Enrolled',
          'Number of ED Applicants', 'Number of ED Applicants Admitted',
          'EA Deadline', 'EA Notification', 'ED I Deadline',
          'ED I Notification', 'ED II Deadline', 'ED II Notification',
          'Average GPA', 'SAT Math 25%', 'SAT Math 75%', 'SAT Reading 25%',
          'SAT Reading 75%', 'GPA >3.75', 'GPA >3.50', 'GPA >3.25',
          'GPA >3.00', 'GPA >2.50', 'GPA >2.00', 'GPA >1.00 ',
          'SAT Math (200-299)', 'SAT Math (700-800)', 'SAT Math (600-699)',
          'SAT Math (500-599)', 'SAT Math (400-499)', 'SAT Math (300-399)',
          'SAT Reading (700-800)', 'SAT Reading (600-699)',
          'SAT Reading (500-599)', 'SAT Reading (400-499)',
          'SAT Reading (300-399)', 'SAT Reading (200-299)', 'filename')
      if column in results.columns
  ]
  results = results[OUTPUT_COLUMNS]
  results.to_csv('results.csv')


if __name__ == '__main__':
  main()