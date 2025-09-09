import logging
import threading
from typing import Optional

import make87 as m87
import mcp
import zenoh.handlers
from make87.interfaces.zenoh import ZenohInterface
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from ollama import Client, Message, Image

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = "Describe the content of the image in detail. Also describe positions of objects and their distance relative to the image center."

class ImageAnalyzer:
    def __init__(self, model: str = "gemma3"):
        self.client = Client()
        self.model = model
        self._last_image = None
        self._lock = threading.RLock()

        server = mcp.server.FastMCP(name="image_describer", host="0.0.0.0", port=9988)
        self.server = server
        @server.tool(
            description="Describe the latest camera image using the given prompt (defaults to a general description prompt)."
        )
        def get_camera_image_description(prompt: str = DEFAULT_PROMPT) -> str:
            if self._last_image is not None:
                with self._lock:
                    cloned_bytes = bytes(self._last_image)
                return self.describe_image(image=cloned_bytes, prompt=prompt)
            else:
                return "no image seen yet."

        self._server_thread = None

    def describe_image(self, image: bytes, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                Message(
                    role="user",
                    content=prompt,
                    images=[Image(
                        value=image
                    )]
                )
            ],
            options={"temperature": 0},  # Set temperature to 0 for more deterministic output
        )
        return response.message.content

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
                    with self._lock:
                        self._last_image = image.data
            except Exception as e:
                logger.error(f"Error in main loop: {e}")


def main():
    logging.basicConfig(level=logging.INFO)
    analyzer = ImageAnalyzer(model="gemma3")
    analyzer.run()


if __name__ == "__main__":
    main()
