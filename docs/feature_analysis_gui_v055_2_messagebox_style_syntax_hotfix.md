# v0.55.2 MessageBox Style Syntax Hotfix

Fixes a stylesheet escaping bug introduced in v0.55.1. Qt stylesheet blocks inside the Python f-string now use escaped braces, preventing `NameError: name 'background' is not defined` during GUI startup.

No analysis logic, tables, plots, recommendations, or export outputs are changed.
