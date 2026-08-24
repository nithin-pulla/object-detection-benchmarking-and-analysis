"""Shared benchmark-controlled components used by all three model notebooks.

Introduced as a P0 fix for the fairness/validity gaps identified in the
repo audit: mixed evaluation implementations, an asymmetric speed-timing
methodology, and duplicated dataset-conversion logic. Every notebook
imports this package rather than re-implementing these pieces locally.
"""
