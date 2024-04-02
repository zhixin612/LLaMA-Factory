from llmtuner import create_ui


def main():
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True, server_port=8888)


if __name__ == "__main__":
    main()
