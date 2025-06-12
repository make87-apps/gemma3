import logging

import mcp
from make87.interfaces.zenoh import ZenohInterface
from make87_messages.core.empty_pb2 import Empty
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87.encodings import ProtobufEncoder
import make87 as m87
from ollama import chat, Client, Message, Image

logger = logging.getLogger(__name__)


def main():
    config = m87.config.load_config_from_env()
    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    image_getter = zenoh_interface.get_requester("GET_IMAGE")
    client = Client()

    server = mcp.server.FastMCP(name="image_describer", host="0.0.0.0", port=9988)

    @server.tool(description="Ask a question with respect to the current image captured by the camera.")
    def ask_question_about_camera_frame(message: str) -> str:

        try:
            response = image_getter.get(payload=ProtobufEncoder(message_type=Empty).encode(Empty()))
            for r in response:
                if r.ok is not None:
                    image = ProtobufEncoder(message_type=ImageJPEG).decode(r.ok.payload.to_bytes())

                    response = client.chat(
                        model="gemma3",
                        messages=[
                            Message(
                                role="user",
                                content=message,
                                images=[Image(
                                    value=image.data
                                )]
                            )
                        ],
                        options={"temperature": 0},  # Set temperature to 0 for more deterministic output
                    )
                    return response.message.content
                else:
                    return str(r.err)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing message: {e}"


    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
