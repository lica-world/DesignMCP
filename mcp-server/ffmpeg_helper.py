import logging

import ffmpeg

logger = logging.getLogger(__name__)


HDR_VIDEO_SPECIFICATION = {
    "video_codec": ["hevc"],
    "pix_fmt": ["yuv420p10le"],
    "color_space": ["bt2020nc"],
}


class VideoMetadataReader:
    @staticmethod
    def get_default_metadata() -> dict:
        """
        This method returns the default metadata of a video file.
        """
        return {
            "video_codec": "",
            "audio_codec": "",
            "width": 0,
            "height": 0,
            "duration_video": 0,
            "duration_audio": 0,
            "rotation": 0,
            "is_HDR": False,
            "has_alpha": False,
            "video_bitrate": 0,
            "audio_bitrate": 0,
        }

    @staticmethod
    def is_thumbnail_stream(stream: dict) -> bool:
        """
        This method checks if the stream is a thumbnail stream.
        Doing this to avoid considering the thumbnail stream as a video stream.
        """
        decision = False
        disposition = stream.get("disposition", {})
        avg_frame_rate = stream.get("avg_frame_rate")
        disposition_condition = int(disposition.get("attached_pic", "0")) == 1
        avg_frame_rate_condition = avg_frame_rate == "0/0"
        decision = disposition_condition and avg_frame_rate_condition
        return decision

    @staticmethod
    def read_rotation_metadata(video_stream) -> int:
        """
        This method reads the rotation metadata of a video stream.
        Rotation information is usually stored in side_data_list metadata.
        video_stream: dict: 'video stream' metadata.
        """
        rotation = 0
        side_data_list = video_stream.get("side_data_list") or []
        for item in side_data_list:
            sdl_item = item.get("side_data_type") or ""
            if sdl_item == "Display Matrix":
                rotation = item.get("rotation")
                if isinstance(rotation, str):
                    rotation = int(rotation)
                    break
        return int(rotation)

    @staticmethod
    def is_hdr_video(video_stream: dict) -> bool:
        """
        This method checks if the video stream is HDR.
        """
        decision = False
        codec = video_stream.get("codec_name", "").lower()
        pix_fmt = video_stream.get("pix_fmt", "").lower()
        color_space = video_stream.get("color_space", "").lower()
        decision = (
            codec in HDR_VIDEO_SPECIFICATION["video_codec"]
            and pix_fmt in HDR_VIDEO_SPECIFICATION["pix_fmt"]
            and color_space in HDR_VIDEO_SPECIFICATION["color_space"]
        )
        return decision

    @staticmethod
    def has_alpha(video_stream: dict) -> bool:
        """
        This method checks if the video stream has alpha channel.
        """
        pix_fmt = video_stream.get("pix_fmt", "").lower()
        tags = video_stream.get("tags", {})
        case1 = tags.get("alpha_mode", "0")
        case2 = tags.get("ALPHA_MODE", "0")
        alpha_tag = int(case1) or int(case2)
        decision = "a" in pix_fmt or alpha_tag == 1
        return decision

    @staticmethod
    def read_metadata(video_path: str) -> dict:
        """
        This method reads the metadata of a video file and returns it as a dictionary.
        video_path: str: local path or public URL to the video file.
        """
        metadata = VideoMetadataReader.get_default_metadata()

        media_info = ffmpeg.probe(video_path)
        for stream in media_info["streams"]:
            if stream["codec_type"].lower() == "video":
                if VideoMetadataReader.is_thumbnail_stream(stream):
                    continue
                metadata["video_codec"] = stream["codec_name"].lower()
                metadata["width"] = int(stream["width"])
                metadata["height"] = int(stream["height"])
                metadata["duration_video"] = float(stream["duration"])
                metadata["rotation"] = VideoMetadataReader.read_rotation_metadata(
                    stream
                )
                metadata["is_HDR"] = VideoMetadataReader.is_hdr_video(stream)
                metadata["has_alpha"] = VideoMetadataReader.has_alpha(stream)
                if "bit_rate" in stream:
                    metadata["video_bitrate"] = int(stream["bit_rate"])

            elif stream["codec_type"] == "audio":
                if "codec_name" not in stream:
                    continue
                metadata["audio_codec"] = stream["codec_name"].lower()
                if "duration" in stream:
                    metadata["duration_audio"] = float(stream["duration"])
                if "bit_rate" in stream:
                    metadata["audio_bitrate"] = int(stream["bit_rate"])

        return metadata
