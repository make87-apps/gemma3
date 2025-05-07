import logging
from datetime import datetime, timezone

from make87_messages.core.empty_pb2 import Empty
from make87_messages.core.header_pb2 import Header
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.text.text_plain_pb2 import PlainText
import make87
from ollama import chat, Client, Message, Image

logger = logging.getLogger(__name__)


def main():
    make87.initialize()
    endpoint = make87.get_provider(
        name="IMAGE_QUESTION", requester_message_type=PlainText, provider_message_type=PlainText
    )
    image_getter = make87.get_requester(
        name="GET_IMAGE", requester_message_type=Empty, provider_message_type=ImageJPEG
    )
    client = Client()

    def callback(message: PlainText) -> PlainText:

        try:
            image = image_getter.request(message=Empty(), timeout=30)

            response = client.chat(
                model="gemma3",
                messages=[
                    Message(
                        role="user",
                        content=message.body,
                        images=[Image(
                            value=image.data
                        )]
                    )
                ],
                options={"temperature": 0},  # Set temperature to 0 for more deterministic output
            )
            return PlainText(
                header=make87.header_from_message(Header, message=message, append_entity_path="response"),
                body=response.message.content,
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return PlainText(
                header=make87.header_from_message(Header, message=message, append_entity_path="response"),
                body=f"Error processing message: {e}",
            )

    endpoint.provide(callback)
    make87.loop()


if __name__ == "__main__":
    main()
