"""
Shotstack Helper Module

This module provides functionality for adding subtitles to videos using the Shotstack API.
"""

import json
import logging
import os
import subprocess
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import srt
from shotstack_sdk import ApiClient, Configuration
from shotstack_sdk.api.edit_api import EditApi
from shotstack_sdk.model.clip import Clip
from shotstack_sdk.model.edit import Edit
from shotstack_sdk.model.offset import Offset
from shotstack_sdk.model.output import Output
from shotstack_sdk.model.timeline import Timeline
from shotstack_sdk.model.title_asset import TitleAsset
from shotstack_sdk.model.track import Track
from shotstack_sdk.model.video_asset import VideoAsset
from ffmpeg_helper import VideoMetadataReader

logger = logging.getLogger(__name__)


class ShotstackSubtitleProcessor:
    """Handle subtitle processing using Shotstack API."""

    def __init__(self, api_key: str, environment: str = "v1"):
        """Initialize the Shotstack client.

        Args:
            api_key: Shotstack API key
            environment: "stage" for testing or "v1" for production
        """
        self.api_key = api_key
        self.environment = environment

        # Configure Shotstack client with proper SSL handling
        host_url = f"https://api.shotstack.io/{self.environment}"
        configuration = Configuration(host=host_url)
        configuration.api_key["DeveloperKey"] = api_key

        # Configure SSL settings
        import ssl

        import certifi

        configuration.verify_ssl = True
        configuration.ssl_ca_cert = certifi.where()  # Use proper CA certificate bundle
        configuration.cert_file = None
        configuration.key_file = None

        # Additional SSL context configuration
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass

        self.api_client = ApiClient(configuration)
        self.api_instance = EditApi(self.api_client)

    def parse_srt_content(self, srt_content: str) -> List[Any]:
        """Parse SRT subtitle content.

        Args:
            srt_content: Raw SRT file content as string

        Returns:
            List of parsed subtitle objects
        """
        try:
            subtitles = list(srt.parse(srt_content))
            return subtitles
        except Exception as e:
            raise ValueError(f"Failed to parse SRT content: {str(e)}")

    def create_subtitles_from_text(
        self, subtitle_text: str, video_duration: float
    ) -> List[Any]:
        """Create simple subtitles from plain text.

        Args:
            subtitle_text: Plain text to be shown as subtitles
            video_duration: Duration of the video in seconds

        Returns:
            List of subtitle objects
        """
        # Split text into chunks (roughly 50 characters per subtitle)
        words = subtitle_text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk + " " + word) <= 50:
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        # Create subtitle objects with timing
        subtitles = []
        duration_per_chunk = video_duration / len(chunks) if chunks else video_duration

        for i, chunk in enumerate(chunks):
            start_time = timedelta(seconds=float(i * duration_per_chunk))
            end_time = timedelta(seconds=float((i + 1) * duration_per_chunk))

            # Create a simple subtitle object (mimicking srt structure)
            subtitle = type(
                "Subtitle",
                (),
                {
                    "content": chunk,
                    "start": start_time,
                    "end": end_time,
                    "index": i + 1,
                },
            )()

            subtitles.append(subtitle)

        return subtitles

    def create_subtitle_clips(self, subtitles: List[Any], max_chars_per_clip: int = 20) -> List[Clip]:
        """Create Shotstack subtitle clips using TitleAsset with character limiting.

        Args:
            subtitles: List of parsed subtitle objects
            max_chars_per_clip: Preferred maximum characters per clip (may be exceeded for long single words)

        Returns:
            List of Shotstack Clip objects
        """
        clips = []

        for subtitle in subtitles:
            start = float(subtitle.start.total_seconds())
            end = float(subtitle.end.total_seconds())
            duration = float(end - start)

            logger.info(f"Processing original subtitle: '{subtitle.content}' from {start:.3f}s to {end:.3f}s (duration: {duration:.3f}s)")

            # Split long subtitles into multiple clips if needed
            subtitle_clips = self._create_char_limited_clips(
                subtitle.content, start, duration, max_chars_per_clip
            )
            clips.extend(subtitle_clips)

        return clips
    
    def _create_char_limited_clips(self, text: str, start_time: float, duration: float, max_chars: int) -> List[Clip]:
        """Create clips from text with character limit while never breaking words.
        
        Rules:
        - If subtitle is empty and word exceeds limit: Include word anyway (preserves word integrity)
        - If subtitle has content and adding word would exceed limit: Move word to next subtitle
        - Words are never split or broken
        
        Args:
            text: Subtitle text content
            start_time: Start time in seconds
            duration: Total duration in seconds
            max_chars: Preferred maximum characters per clip (may be exceeded for long single words)
            
        Returns:
            List of clips (may be multiple if text is split)
        """
        # If text is within limit, create single clip
        if len(text) <= max_chars:
            logger.info(f"Creating single subtitle clip: '{text}' ({len(text)} chars) from {start_time:.3f}s to {start_time + duration:.3f}s (duration: {duration:.3f}s)")
            
            title_asset = TitleAsset(
                text=text,
                style="subtitle",
                color="#ffffff",
                size="x-small", 
                position="bottom",
                background="#000000"
            )
            
            clip = Clip(
                asset=title_asset,
                start=float(start_time),
                length=float(duration)
            )
            
            logger.debug(f"Created single clip object: start={float(start_time)}, length={float(duration)}")
            return [clip]
        
        # Split long text into multiple clips while preserving complete words
        clips = []
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if not current_chunk:
                # Current subtitle is empty - always add the word, even if it exceeds limit
                current_chunk = word
                if len(word) > max_chars:
                    logger.info(f"Long word '{word}' ({len(word)} chars) added to empty subtitle (exceeds {max_chars} limit but preserving word)")
            else:
                # Current subtitle has content - check if adding word would exceed limit
                test_chunk = current_chunk + " " + word
                if len(test_chunk) <= max_chars:
                    # Adding this word stays within limit
                    current_chunk = test_chunk
                else:
                    # Adding this word would exceed limit - move to next subtitle
                    chunks.append(current_chunk)
                    current_chunk = word
                    if len(word) > max_chars:
                        logger.info(f"Long word '{word}' ({len(word)} chars) moved to new subtitle (exceeds {max_chars} limit but preserving word)")
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Calculate duration per chunk
        chunk_duration = duration / len(chunks) if chunks else duration
        
        # Create clips for each chunk
        logger.info(f"Splitting subtitle into {len(chunks)} chunks. Original timing: {start_time}s to {start_time + duration}s (duration: {duration}s)")
        
        for i, chunk in enumerate(chunks):
            chunk_start = start_time + (i * chunk_duration)
            chunk_end = chunk_start + chunk_duration
            
            logger.info(f"Creating subtitle chunk {i+1}/{len(chunks)}: '{chunk}' ({len(chunk)} chars) from {chunk_start:.3f}s to {chunk_end:.3f}s")
            
            title_asset = TitleAsset(
                text=chunk,
                style="subtitle",
                color="#ffffff", 
                size="x-small",
                position="bottom",
                background="#000000"
            )
            
            clip = Clip(
                asset=title_asset,
                start=float(chunk_start),
                length=float(chunk_duration)
            )
            
            logger.debug(f"Created clip object: start={float(chunk_start)}, length={float(chunk_duration)}")
            clips.append(clip)
        
        return clips


    def get_video_properties(self, video_path):
        """Get video properties. For URLs, use smart detection instead of ffprobe."""
        try:
            metadata = VideoMetadataReader.read_metadata(video_path)  # To validate video
            width = metadata["width"]
            height = metadata["height"]
            if metadata["rotation"] != 0:
                width, height = height, width  # Swap if rotated

            # Aspect ratio as simplified string
            if width > height:
                aspect = "16:9"
            elif width < height:
                aspect = "9:16"
            else:
                aspect = "1:1"

            # Map resolution
            if width >= 3840:
                resolution = "uhd"
            elif width >= 1920:
                resolution = "hd"  
            elif width >= 1280:
                resolution = "hd"
            else:
                resolution = "sd"

            return {
                "width": width,
                "height": height,
                "aspect_ratio": aspect,
                "resolution": resolution,
            }
            
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            return self._get_default_video_properties()
    
    def _get_default_video_properties(self):
        """Return sensible default video properties."""
        return {
            "width": 1920,
            "height": 1080, 
            "aspect_ratio": "16:9",
            "resolution": "hd",
        }
    

    def add_subtitles_to_video(
        self,
        video_url: str,
        subtitles: List[Any],
    ) -> str:
        """Add subtitles to a video using Shotstack.

        Args:
            video_url: URL of the source video
            subtitles: List of parsed subtitle objects

        Returns:
            Render ID for tracking the job
        """

        if not subtitles:
            raise ValueError("No subtitles provided")
        output_format = "mp4"

        # Auto-detect video properties if not specified
        video_properties = self.get_video_properties(video_url)
        logger.info(f"Video properties: {video_properties}")

        max_chars_per_subtitle = 10 if video_properties["width"] <= video_properties["height"] else 30

        # Use detected properties or fallback to provided values
        resolution = video_properties.get("resolution", "hd")

        # Use improved TitleAsset approach with character limiting
        subtitle_clips = self.create_subtitle_clips(subtitles, max_chars_per_clip=max_chars_per_subtitle)
        logger.info(f"Created {len(subtitle_clips)} subtitle clips using enhanced TitleAsset format (preferred {max_chars_per_subtitle} chars per clip, words never broken)")

        # Calculate video duration from last subtitle
        video_duration = subtitles[-1].end.total_seconds()
        logger.info(f"Final video duration based on last subtitle: {video_duration:.3f}s")
        logger.info(f"Total subtitle clips created: {len(subtitle_clips)}")

        # Create background video clip
        video_asset = VideoAsset(src=video_url)
        video_clip = Clip(asset=video_asset, start=0.0, length=video_duration)
        logger.debug(f"Created background video clip: start=0.0, length={video_duration}")

        # Create tracks - matching working JSON structure (subtitle track first)
        subtitle_track = Track(clips=subtitle_clips)
        video_track = Track(clips=[video_clip])
        
        # Debug: Show timing summary of all subtitle clips
        logger.info("=== SUBTITLE TIMING SUMMARY ===")
        for i, clip in enumerate(subtitle_clips):
            clip_end = clip.start + clip.length
            logger.info(f"Clip {i+1}: {clip.start:.3f}s to {clip_end:.3f}s (length: {clip.length:.3f}s)")
        logger.info("=== END TIMING SUMMARY ===")

        # Create timeline with subtitle track first (matching working JSON)
        timeline = Timeline(
            background="#000000",  # Black background as in working JSON
            tracks=[subtitle_track, video_track]  # Subtitle track first
            # No soundtrack parameter - let original video audio pass through
        )

        # Create output settings using explicit dimensions like working JSON
        width = video_properties.get("width", 1280)
        height = video_properties.get("height", 720)
        
        output = Output(
            format=output_format,  # Matches input video format
            resolution=resolution,  # Matches input video resolution
            fps=float(25),                # Standard FPS as in working JSON
            aspectRatio=video_properties.get("aspect_ratio", "16:9"),
            # Note: Shotstack SDK might handle size differently than raw JSON
        )

        # Create edit
        edit = Edit(timeline=timeline, output=output)

        # Submit render request
        try:
            logger.info(f"Submitting Shotstack render with {len(subtitle_clips)} subtitle clips")
            logger.info(f"Video duration: {video_duration}s, Output: {output_format} {resolution}")
            
            response = self.api_instance.post_render(edit)
            render_id = response.response.id
            logger.info(f"Shotstack render submitted successfully with ID: {render_id}")
            return render_id
        except Exception as e:
            logger.error(f"Shotstack render submission failed: {str(e)}")
            logger.error(f"Edit object: timeline tracks={len(edit.timeline.tracks)}, output={edit.output}")
            raise RuntimeError(f"Failed to submit render request: {str(e)}")

    def get_render_status(self, render_id: str) -> Dict[str, Any]:
        """Get the status of a render job.

        Args:
            render_id: The render job ID

        Returns:
            Dictionary containing render status and details
        """
        try:
            response = self.api_instance.get_render(render_id)
            return {
                "id": response.response.id,
                "status": response.response.status,
                "url": response.response.url,
                "error": response.response.error,
                "created": response.response.created,
                "updated": response.response.updated,
            }
        except Exception as e:
            # Handle SDK validation errors for newer status values like "preprocessing"
            if "Invalid value for `status`" in str(e) and "preprocessing" in str(e):
                # Make raw API call to get status when SDK validation fails
                import requests

                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                }

                url = f"https://api.shotstack.io/{self.environment}/render/{render_id}"

                try:
                    raw_response = requests.get(url, headers=headers, timeout=30)
                    if raw_response.status_code == 200:
                        data = raw_response.json()
                        response_data = data.get("response", {})

                        return {
                            "id": response_data.get("id"),
                            "status": response_data.get(
                                "status"
                            ),  # This will include "preprocessing"
                            "url": response_data.get("url"),
                            "error": response_data.get("error"),
                            "created": response_data.get("created"),
                            "updated": response_data.get("updated"),
                        }
                    else:
                        raise RuntimeError(
                            f"Raw API call failed: {raw_response.status_code} - {raw_response.text}"
                        )
                except requests.RequestException as req_e:
                    raise RuntimeError(
                        f"Failed to get render status via raw API: {str(req_e)}"
                    )
            else:
                raise RuntimeError(f"Failed to get render status: {str(e)}")

    def wait_for_render_completion(
        self, render_id: str, max_wait_time: int = 600, poll_interval: int = 10
    ) -> Dict[str, Any]:
        """Wait for render completion and return final status.

        Args:
            render_id: The render job ID
            max_wait_time: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds

        Returns:
            Dictionary containing final render status and details
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status = self.get_render_status(render_id)

            if status["status"] == "done":
                return status
            elif status["status"] == "failed":
                error_msg = status.get("error", "Unknown error")
                raise RuntimeError(f"Render failed: {error_msg}")
            elif status["status"] in [
                "queued",
                "fetching",
                "rendering",
                "saving",
                "preprocessing",
            ]:
                # Valid processing statuses - continue waiting
                logger.info(
                    f"Render status: {status['status']} - continuing to wait..."
                )
                pass
            else:
                # Unknown status - log it but continue waiting
                logger.warning(
                    f"Unknown render status: {status['status']} - continuing to wait..."
                )

            time.sleep(poll_interval)

        raise TimeoutError(f"Render timed out after {max_wait_time} seconds")


def get_shotstack_api_key() -> Optional[str]:
    """Get the Shotstack API key from environment variables.

    Returns:
        The Shotstack API key if available, None otherwise.
    """
    return os.getenv("SHOTSTACK_API_KEY")


def get_video_duration_estimate(subtitles: List[Any]) -> float:
    """Estimate video duration from subtitles.

    Args:
        subtitles: List of parsed subtitle objects

    Returns:
        Estimated duration in seconds
    """
    if not subtitles:
        return 0.0

    return subtitles[-1].end.total_seconds()


class ElevenLabsTranscriber:
    """Handle audio/video transcription using ElevenLabs API."""

    def __init__(self, api_key: str):
        """Initialize the ElevenLabs transcriber.

        Args:
            api_key: ElevenLabs API key
        """
        self.api_key = api_key

    def transcribe_file_to_srt(self, file_path: str) -> Optional[str]:
        """Transcribe audio or video file to SRT format using ElevenLabs SDK.

        Args:
            file_path: Path to audio or video file

        Returns:
            SRT content as string, or None if failed
        """
        try:
            from elevenlabs import ElevenLabs

            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=self.api_key)

            # Transcribe from local file using SDK
            with open(file_path, "rb") as audio_file:
                transcript = client.speech_to_text.convert(
                    model_id="scribe_v1", file=audio_file
                )

            # Return the SRT format directly from SDK, or convert using word timing data
            if hasattr(transcript, "srt") and transcript.srt:
                return transcript.srt
            elif hasattr(transcript, "words") and transcript.words:
                # Convert using precise word timing data
                return self._convert_words_to_srt(transcript.words)
            elif hasattr(transcript, "text") and transcript.text:
                # Fallback to basic text-to-SRT conversion
                return self._convert_text_to_srt(transcript.text)
            else:
                print("No transcript text or words available")
                return None

        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def transcribe_url_to_srt(self, media_url: str) -> Optional[str]:
        """Transcribe audio or video file from URL to SRT format using ElevenLabs SDK.

        Args:
            media_url: URL to audio or video file

        Returns:
            SRT content as string, or None if failed
        """
        try:
            from elevenlabs import ElevenLabs

            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=self.api_key)

            # Transcribe from cloud URL using SDK
            transcript = client.speech_to_text.convert(
                model_id="scribe_v1", cloud_storage_url=media_url
            )
            # logger.info(f"Transcript: {transcript}")

            # Return the SRT format directly from SDK, or convert using word timing data
            if hasattr(transcript, "srt") and transcript.srt:
                return transcript.srt
            elif hasattr(transcript, "words") and transcript.words:
                # Convert using precise word timing data
                return self._convert_words_to_srt(transcript.words)
            elif hasattr(transcript, "text") and transcript.text:
                # Fallback to basic text-to-SRT conversion
                return self._convert_text_to_srt(transcript.text)
            else:
                print("No transcript text or words available")
                return None

        except Exception as e:
            print(f"Error during URL transcription: {e}")
            return None

    def transcribe_video_to_srt(self, video_file_or_url: str) -> Optional[str]:
        """Transcribe video file or URL directly to SRT format.

        Args:
            video_file_or_url: Path to video file or URL

        Returns:
            SRT content as string, or None if failed
        """
        # Check if it's a URL or local file
        if video_file_or_url.startswith(("http://", "https://")):
            return self.transcribe_url_to_srt(video_file_or_url)
        else:
            return self.transcribe_file_to_srt(video_file_or_url)

    def _convert_words_to_srt(self, words: list) -> str:
        """Convert ElevenLabs word timing data to SRT format.

        Args:
            words: List of word objects with timing data from ElevenLabs

        Returns:
            SRT formatted string
        """
        if not words:
            return ""

        srt_lines = []
        segment_number = 1
        current_segment_words = []
        current_segment_start = None

        # Filter out spacing entries and collect only actual words
        word_entries = [w for w in words if hasattr(w, "type") and w.type == "word"]

        for i, word in enumerate(word_entries):
            # Get word text and timing
            word_text = getattr(word, "text", "")
            word_start = getattr(word, "start", 0)
            word_end = getattr(word, "end", word_start + 1)

            if current_segment_start is None:
                current_segment_start = word_start

            current_segment_words.append(word_text)

            # Create new segment when we have enough words, time has passed, or at sentence boundaries
            segment_duration = word_end - current_segment_start
            is_sentence_end = word_text.rstrip().endswith((".", "!", "?"))
            should_break = (
                len(current_segment_words) >= 8  # Max 8 words per subtitle
                or segment_duration >= 3.5  # Max 3.5 seconds per subtitle
                or is_sentence_end  # Break at sentence boundaries
                or i == len(word_entries) - 1  # Last word
            )

            if should_break and current_segment_words:
                # Format timestamps
                start_time = self._seconds_to_srt_time(current_segment_start)
                end_time = self._seconds_to_srt_time(word_end)

                # Create SRT entry
                subtitle_text = " ".join(current_segment_words).strip()
                srt_entry = (
                    f"{segment_number}\n{start_time} --> {end_time}\n{subtitle_text}\n"
                )
                srt_lines.append(srt_entry)

                # Reset for next segment
                segment_number += 1
                current_segment_words = []
                current_segment_start = None

        return "\n".join(srt_lines)

    def _convert_text_to_srt(self, text: str) -> str:
        """Convert plain text to basic SRT format.

        Args:
            text: Plain text transcript

        Returns:
            SRT formatted string
        """
        if not text or not text.strip():
            return ""

        # Split text into sentences (basic approach)
        import re

        sentences = re.split(r"[.!?]+", text)

        srt_lines = []
        segment_number = 1
        current_time = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Estimate duration based on sentence length (roughly 0.5 seconds per word)
            word_count = len(sentence.split())
            duration = max(2, word_count * 0.5)  # Minimum 2 seconds

            start_time = self._seconds_to_srt_time(current_time)
            end_time = self._seconds_to_srt_time(current_time + duration)

            srt_entry = f"{segment_number}\n{start_time} --> {end_time}\n{sentence}\n"
            srt_lines.append(srt_entry)

            segment_number += 1
            current_time += duration

        return "\n".join(srt_lines)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            SRT formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

