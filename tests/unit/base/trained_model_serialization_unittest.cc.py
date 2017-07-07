#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

# Classes that need custom initialization
IGNORE_LIST = [
    'CFeatureBlockLogisticRegression', 'CLibLinearMTL',
    'CMultitaskLinearMachine', 'CMultitaskLogisticRegression',
    'CMultitaskL12LogisticRegression', 'CMultitaskLeastSquaresRegression',
    'CMultitaskTraceLogisticRegression', 'CMultitaskClusteredLogisticRegression',
    'CLatentSVM', 'CLatentSOSVM',
    'CDomainAdaptationSVMLinear',
]

def read_defined_guards(config_file):
    with open(config_file) as f:
        config = f.read().lower()
        return re.findall('#define (\w+)', config)


def is_guarded(include, defined_guards):
    with open(include) as header:
        guards = re.findall('#ifdef (\w+)', header.read().lower())
        return any([g not in defined_guards for g in guards])


def ignore_in_class_list(include):
    with open(include) as header:
        return 'IGNORE_IN_CLASSLIST' in header.read()


def is_pure_virtual(name, tags):
    return any([name + '\timplementation:pure virtual' in tag for tag in tags])


def is_shogun_class(c):
    return c[0] == 'C' and c[1].isupper() and 'class' in c


def get_shogun_classes(tags):
    classes = {}
    # in ctags format it is TAG\tLOCATION\t..\tinherits:CLASS
    for line in filter(is_shogun_class, tags):
        attrs = line.strip().split('\t')
        inherits_str = 'inherits:'
        symbol, location = attrs[0], attrs[1]
        base = attrs[-1][len(inherits_str):] if attrs[-1].startswith(inherits_str) else None
        classes[symbol] = {
            'include': location,
            'base': base}
    return classes


def get_ancestors(classes, name):
    b = classes[name]['base']
    return [b] + get_ancestors(classes, b) if b in classes else []


def read_ctags(filename):
    if not os.path.exists(filename):
        raise Exception('Failed to found ctags file at %s' % (filename))
    with open(filename) as file:
        return file.readlines()


def entry(templateFile, input_file, config_file):
    templateLoader = jinja2.FileSystemLoader(searchpath="/")
    templateEnv = jinja2.Environment(loader=templateLoader)

    template = templateEnv.get_template(templateFile)

    tags = read_ctags(input_file)
    classes = get_shogun_classes(tags)
    guards = read_defined_guards(config_file)

    # Get all linear machines
    global IGNORE_LIST
    base = 'CLinearMachine'
    machines = {name: attrs for name, attrs in classes.items() if name not in IGNORE_LIST and
                base in get_ancestors(classes, name) and not is_guarded(attrs['include'], guards)
                and not is_pure_virtual(name, tags) and not ignore_in_class_list(attrs['include'])}

    templateVars = {"classes" : machines}

    return template.render(templateVars)


# execution
# ./trained_model_serialization_unittest.cc.py
# <template file> <input file> <output file> <config file>

import sys, os, re, jinja2
TEMPLATE_FILE = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]
config_file = sys.argv[4]

outputText = entry(TEMPLATE_FILE, input_file, config_file)

with open(output_file, 'w') as f:
    f.writelines(outputText)
