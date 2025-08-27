import streamlit as st
import pandas as pd
import pytz
import subprocess
from datetime import timedelta



class helpers:
    def timestring_to_seconds(timestring):
        if pd.isna(timestring) or timestring == '0' or timestring == 0 or timestring.strip() == '':
            return 0

        if isinstance(timestring, float):
            timestring = str(int(timestring)) 

        timestring = str(timestring).strip()

        if 'T' in timestring:
            days_part, time_part = timestring.split('T')
        else:
            days_part, time_part = '0', timestring

        try:
            days = int(days_part.strip()) if days_part.strip() else 0
        except ValueError:
            days = 0

        # Convert time part (HH:MM:SS)
        time_parts = time_part.split(':')
        try:
            hours = int(time_parts[0].strip()) if len(time_parts) > 0 and time_parts[0].strip() else 0
        except ValueError:
            hours = 0
        try:
            minutes = int(time_parts[1].strip()) if len(time_parts) > 1 and time_parts[1].strip() else 0
        except ValueError:
            minutes = 0
        try:
            seconds = int(time_parts[2].strip()) if len(time_parts) > 2 and time_parts[2].strip() else 0
        except ValueError:
            seconds = 0

        # Calculate total seconds
        total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
        return total_seconds



    def seconds_to_timestring(total_seconds):
        if pd.isna(total_seconds):
            return None
        if isinstance(total_seconds, float):
            total_seconds = int(total_seconds)

        if isinstance(total_seconds, int) and total_seconds >= 0: 

            td = timedelta(seconds=total_seconds)

            days = td.days
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            seconds = round(seconds)  

            timestring = f"{days}T {hours}:{minutes}:{seconds}"
            return timestring
        else:
            return None

    def format_interval_label(interval):
        min_time = interval.left
        max_time = interval.right

        def format_time(minutes):
            days = int(minutes // 1440)  
            hours = int((minutes % 1440) // 60)
            mins = int(minutes % 60)

            if days > 0 and hours > 0:
                return f"{days}d {hours}h"
            elif days > 0:
                return f"{days}d"
            elif hours > 0 and mins > 0:
                return f"{hours}h {mins}m"
            elif hours > 0:
                return f"{hours}h"
            else:
                return f"{mins}m"

        min_time_str = format_time(min_time)
        max_time_str = format_time(max_time)
        return f"{min_time_str} - {max_time_str}"


    def get_job_script(jobid):
        command = ["sacct", "-B", "-j", jobid]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            st.code(output)
        except subprocess.CalledProcessError as e:
            st.error(f"Error details: {e}")
            

    def readable_with_commas(value):
        if value >= 100_000:
            return f"{value / 1_000_000:.3f}M"
        elif value <= 100_000 and value >= 1_000:
            return f"{value / 1_000:.3f}K"
        else:
            return f"{value:,}"

    def convert_timestamps_to_berlin_time(df: pd.DataFrame) -> pd.DataFrame:
            """
            Convert Unix timestamps to Berlin timezone and format as strings.
            
            Args:
                df: DataFrame with 'Start' and 'End' columns containing Unix timestamps
                
            Returns:
                DataFrame with formatted datetime strings
            """

            berlin_tz = pytz.timezone('Europe/Berlin')
            
            # Convert Unix timestamps to datetime and localize to UTC
            df['Start'] = pd.to_datetime(df['Start'], unit='s', errors='coerce').dt.tz_localize('UTC')
            df['End'] = pd.to_datetime(df['End'], unit='s', errors='coerce').dt.tz_localize('UTC')

            # Convert to Berlin time (handles daylight saving time automatically)
            df['Start'] = df['Start'].dt.tz_convert(berlin_tz)
            df['End'] = df['End'].dt.tz_convert(berlin_tz)

            # Format datetime columns without timezone offset
            df['Start'] = df['Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['End'] = df['End'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            return df


    def build_conditions(query, params, partition_selector=None, allowed_groups=None,user_role=None, current_user=None, ):

        if partition_selector:
            placeholders = ','.join(['?'] * len(partition_selector))
            query += f" AND Partition IN ({placeholders})"
            params.extend(partition_selector)


        if current_user:
            query += " AND User = ?"
            params.append(current_user)

        if allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            query += f" AND Account IN ({placeholders})"
            params.extend(allowed_groups)

        return query, params
