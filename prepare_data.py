from pycparser import c_parser, c_ast
import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
import numpy as np


class Tree_Node(object):
    def __init__(self, node):
        self.node = node
        # self.vocab = word_map
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        # self.index = self.token_to_index(self.token)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile','Switch']:
            return [Tree_Node(children[0][1])]
        elif self.token == 'For':
            return [Tree_Node(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [Tree_Node(child) for _, child in children]

    def children(self):
        return self.children
   
class SingleNode(Tree_Node):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token



def get_sequences(node, sequence):
    current = SingleNode(node)
    sequence.append(current.get_token())
    for _, child in node.children():
        get_sequences(child, sequence)
    if current.get_token().lower() == 'compound':
        sequence.append('End')


def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
        block_seq.append(Tree_Node(node))
        if name is not 'For':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                block_seq.append(Tree_Node(child))
            get_blocks(child, block_seq)
    elif name is 'Compound':
        block_seq.append(Tree_Node(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                block_seq.append(Tree_Node(child))
            get_blocks(child, block_seq)
        block_seq.append(Tree_Node('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)
