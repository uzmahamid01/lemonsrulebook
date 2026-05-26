#!/usr/bin/env python3
"""Simple forum search example for tree output format."""

import json

def find_threads_with_keyword(filename, keyword):
    """Find thread IDs containing posts with a specific keyword."""
    thread_ids = set()
    with open(filename) as f:
        for line in f:
            item = json.loads(line)
            if item['type'] == 'post' and keyword in item.get('content', ''):
                thread_ids.add(item['path'][-2])  # Thread ID is second-to-last in path
    return list(thread_ids)

if __name__ == '__main__':
    filename = 'forum_data.json'
    
    # Find threads mentioning specific keywords
    fuel_threads = find_threads_with_keyword(filename, 'fuel')
    harness_threads = find_threads_with_keyword(filename, 'harness')
    
    print(f"Threads mentioning 'fuel': {fuel_threads}")
    print(f"Threads mentioning 'harness': {harness_threads}")
