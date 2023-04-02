#!/usr/bin/evn python3
# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Create release notes with the issues from a milestone.
    python3 release_notes.py -c didi delta v.xxxxx
"""
import argparse
import collections
import json
import sys
import urllib.request

github_url = 'https://api.github.com/repos'

if __name__ == '__main__':
    # usage:
    # 1. close milestone on github
    # 2. python3 tools/release_notes.py -c didi delta v0.3.3

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Create a draft release with the issues from a milestone.',
    )

    parser.add_argument(
        'user',
        metavar='user',
        type=str,
        default='paddlepaddle',
        help='github user: paddlepaddle')

    parser.add_argument(
        'repository',
        metavar='repository',
        type=str,
        default='paddlespeech',
        help='github repository: paddlespeech')

    parser.add_argument(
        'milestone',
        metavar='milestone',
        type=str,
        help='name of used milestone: v0.3.3')

    parser.add_argument(
        '-c',
        '--closed',
        help='Fetch closed milestones/issues',
        action='store_true')

    parser.print_help()
    args = parser.parse_args()

    # Fetch milestone infos
    url = "%s/%s/%s/milestones" % (github_url, args.user, args.repository)

    headers = {
        'Origin':
        'https://github.com',
        'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64) '
        'AppleWebKit/537.11 (KHTML, like Gecko) '
        'Chrome/23.0.1271.64 Safari/537.11',
        'Accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset':
        'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding':
        'none',
        'Accept-Language':
        'en-US,en;q=0.8',
        'Connection':
        'keep-alive'
    }

    if args.closed:
        url += "?state=closed"

    req = urllib.request.Request(url, headers=headers)
    github_request = urllib.request.urlopen(req)
    if not github_request:
        parser.error('Cannot read milestone list.')

    decoder = json.JSONDecoder()
    milestones = decoder.decode(github_request.read().decode('utf-8'))
    github_request.close()

    print('parse milestones', file=sys.stderr)
    milestone_id = None
    for milestone in milestones:
        if milestone['title'] == args.milestone:
            milestone_id = milestone['number']
    if not milestone_id:
        parser.error('Cannot find milestone')

    # Get milestone related issue info
    url = '%s/%s/%s/issues?milestone=%d' % (github_url, args.user,
                                            args.repository, milestone_id)
    if args.closed:
        url += "&state=closed"

    req = urllib.request.Request(url, headers=headers)
    github_request = urllib.request.urlopen(req)
    if not github_request:
        parser.error('Cannot read issue list.')

    issues = decoder.decode(github_request.read().decode('utf-8'))
    github_request.close()

    #print('parse issues', file=sys.stderr)
    #final_data = []
    #labels = []
    #thanks_to = []
    #for issue in issues:

    #  for label in issue['labels']:
    #    labels.append(label['name'])

    #  thanks_to.append('@%s' % (issue['user']['login']))
    #  final_data.append(' * **[%s]** - %s #%d by **@%s**\n' % (
    #    label['name'],
    #    issue['title'],
    #    issue['number'],
    #    issue['user']['login']
    #  ))

    #dic = collections.defaultdict(set)
    #for l_release in list(set(labels)):

    #  for f_data in final_data:
    #    if '[%s]' % l_release in f_data:
    #      dic[l_release].add(f_data)

    #with open(f"release_note_issues_{args.milestone}.md", 'w') as f:
    #  for key, value in dic.items():
    #    print('# %s\n%s' % (key, ''.join(value)), file=f)
    #  print('# %s\n%s' % ('Acknowledgements', 'Special thanks to %s ' % ('  '.join(list(set(thanks_to))))), file=f)

    # Get milestone related PR info
    url = '%s/%s/%s/pulls?milestone=%d' % (github_url, args.user,
                                           args.repository, milestone_id)
    if args.closed:
        url += "&state=closed"

    req = urllib.request.Request(url, headers=headers)
    github_request = urllib.request.urlopen(req)
    if not github_request:
        parser.error('Cannot read issue list.')

    issues = decoder.decode(github_request.read().decode('utf-8'))
    github_request.close()

    print('parse pulls', file=sys.stderr)
    final_data = []
    labels = []
    thanks_to = []
    for issue in issues:
        label = None
        for label in issue['labels']:
            labels.append(label['name'])
            if label:
                thanks_to.append('@%s' % (issue['user']['login']))
                final_data.append(' * **[%s]** - %s #%d by **@%s**\n' %
                                (label['name'], issue['title'], issue['number'],
                                issue['user']['login']))
     
    dic = collections.defaultdict(set)
    for l_release in list(set(labels)):

        for f_data in final_data:
            if '[%s]' % l_release in f_data:
                dic[l_release].add(f_data)

    with open(f"release_note_pulls_{args.milestone}.md", 'w') as f:
        for key, value in dic.items():
            print('# %s\n%s' % (key, ''.join(value)), file=f)
        print(
            '# %s\n%s' % ('Acknowledgements', 'Special thanks to %s ' %
                          ('  '.join(list(set(thanks_to))))),
            file=f)
