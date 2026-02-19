import flet as ft

def main(page: ft.Page):
    try:
        dd = ft.Dropdown(
            label="Test",
            options=[ft.dropdown.Option("1", "One")],
            on_change=lambda e: print("Changed")
        )
        page.add(dd)
        print("Dropdown created successfully")
    except Exception as e:
        print(f"Error creating Dropdown: {e}")

if __name__ == "__main__":
    ft.run(main)
