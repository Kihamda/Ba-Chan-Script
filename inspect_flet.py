import flet as ft
import inspect

print(f"Flet version: {ft.__version__}")
try:
    sig = inspect.signature(ft.Dropdown.__init__)
    print("on_change in init:", 'on_change' in sig.parameters)
    print("on_change in dir:", 'on_change' in dir(ft.Dropdown))
except Exception as e:
    print(f"Error: {e}")
