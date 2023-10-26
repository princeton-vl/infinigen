2.0.0 - Add support for geometry node groups
2.1.0 - Add support for node groups in shader nodes
2.2.0 - Use labels as varname hints, transpile labels, prevent varname clashes with python keywords, use math ops as name hints
2.3.0 - Add support for automatic randomization based on labels
2.3.2 - Use nodegroup node_tree names as variable name hints
2.3.3 - Add type hints on nw arguments
2.4.0 - Add randomization support for Nodes.Integer, add ValueError for commonly missed ATTRS_AVAILABLE
2.4.1 - Add more node_info entries
2.4.2 - Sanitize non alphanumeric/underscore characters from variable names
2.4.3 - Fix duplicated dependencies
2.5.1 - Cleanup function names when transpiling already transpiled code, limit floats to 4 decimal places
2.6.0 - Replace manual NODE_ATTRS_AVAILABLE with automatic func, add aliases for bl3.5.1 nodes, switch to logging library
2.6.1 - Avoid overlength lines, avoid duplicating group_inputs
2.6.2 - Add dependency input, automatically import dependencies we need
2.6.3 - Ignore reroutes
2.6.4 - Fix world/compositor transpiling
2.6.5 - Revert "Ignore reroutes"
