import logging
import threading

import make87 as m87
import mcp
import zenoh.handlers
from make87.interfaces.zenoh import ZenohInterface
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from ollama import Client, Message, Image

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    def __init__(self, model: str = "gemma3"):
        self.client = Client()
        self.model = model
        self.current_prompt = "Describe the content of the image in detail."
        self.current_description = "no image seen yet."

        server = mcp.server.FastMCP(name="image_describer", host="0.0.0.0", port=9988)
        self.server = server
        @server.tool(description="Ask what the current camera image shows with respect to the currently set prompt.")
        def get_camera_image_description() -> str:
            return self.current_description

        @server.tool(description="Ask what the current camera image shows with respect to the currently set prompt.")
        def set_analyzer_prompt(prompt: str) -> str:
            self.current_prompt = prompt
            return f"Set prompt to: {prompt}"

        self._server_thread = None

    def run(self):
        thr = threading.Thread(target=self.server.run, kwargs={"transport": "streamable-http"}, daemon=True)
        thr.start()
        self._server_thread = thr
        zenoh_interface = ZenohInterface(name="zenoh-client")
        subscriber = zenoh_interface.get_subscriber("IMAGE", handler=zenoh.handlers.RingChannel(capacity=1))
        while True:
            try:
                sample = subscriber.recv()
                if sample and sample.payload:
                    image = m87.encodings.ProtobufEncoder(message_type=ImageJPEG).decode(sample.payload.to_bytes())
                    response = self.client.chat(
                        model="gemma3",
                        messages=[
                            Message(
                                role="user",
                                content=self.current_prompt,
                                images=[Image(
                                    value=image.data
                                )]
                            )
                        ],
                        options={"temperature": 0},  # Set temperature to 0 for more deterministic output
                    )
                    self.current_description = response.message.content
            except Exception as e:
                logger.error(f"Error in main loop: {e}")


def main():
    logging.basicConfig(level=logging.INFO)
    analyzer = ImageAnalyzer(model="gemma3")
    analyzer.run()


if __name__ == "__main__":
    main()
