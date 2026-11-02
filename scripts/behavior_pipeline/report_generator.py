
"""Generate an Excel report from civet monitoring pipeline outputs."""

import csv
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import PieChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties


class BehaviorReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "civet_monitoring_report.xlsx"

        self.thin_border = Border(
            left=Side(style="thin", color="BFBFBF"),
            right=Side(style="thin", color="BFBFBF"),
            top=Side(style="thin", color="BFBFBF"),
            bottom=Side(style="thin", color="BFBFBF"),
        )

    # --------------------------------------------------
    # General helpers
    # --------------------------------------------------

    @staticmethod
    def read_csv(path):
        path = Path(path)

        if not path.exists():
            return []

        with open(path, "r", newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))

    @staticmethod
    def normalize_key(value):
        """Normalize CSV column names for flexible matching."""
        return (
            str(value or "")
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("(", "")
            .replace(")", "")
        )

    @classmethod
    def get_value(cls, record, *keys, default=None):
        """Read a CSV value using any of the supplied column names."""
        normalized = {
            cls.normalize_key(key): value
            for key, value in record.items()
        }

        for key in keys:
            value = normalized.get(cls.normalize_key(key))

            if value is not None and str(value).strip() != "":
                return value

        return default

    @staticmethod
    def parse_duration(value):
        """Convert seconds or HH:MM:SS into seconds."""
        if value is None:
            return 0.0

        value = str(value).strip()

        if not value:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        # Support HH:MM:SS or MM:SS values.
        parts = value.split(":")

        try:
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds

            if len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds

        except ValueError:
            pass

        return 0.0

    @classmethod
    def get_record_duration(cls, record):
        """Read duration from common pipeline CSV column names."""
        value = cls.get_value(
            record,
            "duration_seconds",
            "Duration (seconds)",
            "duration",
            "Duration",
            default=0,
        )

        return cls.parse_duration(value)

    @staticmethod
    def format_time(value):
        if value is None:
            return ""

        if isinstance(value, (datetime, time)):
            return value.strftime("%H:%M:%S")

        return str(value)

    @staticmethod
    def format_elapsed(seconds):
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def style_header(cell, fill="1F4E78"):
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.font = Font(color="FFFFFF", bold=True, size=9)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    @staticmethod
    def style_subheader(cell, fill):
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    @staticmethod
    def style_title(sheet, cell_range, title):
        sheet.merge_cells(cell_range)

        cell = sheet[cell_range.split(":")[0]]
        cell.value = title
        cell.font = Font(size=15, bold=True)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

        sheet.row_dimensions[1].height = 28

    def apply_borders(self, sheet, cell_range):
        for row in sheet[cell_range]:
            for cell in row:
                cell.border = self.thin_border

    @staticmethod
    def configure_print(sheet, print_area, header_rows=None):
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(
            fitToPage=True,
            autoPageBreaks=False,
        )

        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.orientation = "portrait"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 1

        sheet.page_margins.left = 0.25
        sheet.page_margins.right = 0.25
        sheet.page_margins.top = 0.35
        sheet.page_margins.bottom = 0.35
        sheet.page_margins.header = 0.15
        sheet.page_margins.footer = 0.15

        sheet.print_options.horizontalCentered = True
        sheet.print_options.verticalCentered = False

        sheet.print_area = print_area

        if header_rows:
            sheet.print_title_rows = header_rows

        sheet.oddFooter.center.text = "Civet Monitoring Report"
        sheet.oddFooter.right.text = "Page &P of &N"

        sheet.sheet_view.showGridLines = True

    # --------------------------------------------------
    # Activity Summary
    # --------------------------------------------------

    def add_activity_sheet(self, workbook, activity_records):
        sheet = workbook.active
        sheet.title = "Activity Summary"

        self.style_title(
            sheet,
            "A1:I1",
            "Civet Activity Summary",
        )

        sheet.merge_cells("A3:C3")
        sheet.merge_cells("D3:F3")
        sheet.merge_cells("G3:I3")

        sheet["A3"] = "Active"
        sheet["D3"] = "Inactive"
        sheet["G3"] = "Not Visible"

        self.style_subheader(sheet["A3"], "ED7D31")
        self.style_subheader(sheet["D3"], "70AD47")
        self.style_subheader(sheet["G3"], "808080")

        for start_column, fill in [
            (1, "ED7D31"),
            (4, "70AD47"),
            (7, "808080"),
        ]:
            for offset, header in enumerate(["No.", "Start", "Stop"]):
                cell = sheet.cell(
                    row=4,
                    column=start_column + offset,
                    value=header,
                )
                self.style_subheader(cell, fill)

        sheet.row_dimensions[3].height = 23
        sheet.row_dimensions[4].height = 25

        intervals = []

        for record in activity_records:
            state = str(record.get("state", "")).strip().lower()

            if state not in {"active", "inactive", "not_visible"}:
                continue

            try:
                elapsed_start = float(
                    self.get_value(
                        record,
                        "start_time",
                        "start_time_seconds",
                        default=float("inf"),
                    )
                )
            except (TypeError, ValueError):
                elapsed_start = float("inf")

            intervals.append({
                "state": state,
                "start": self.format_time(
                    self.get_value(
                        record,
                        "cctv_start_time",
                        "start_timestamp",
                        "start_time",
                        default="",
                    )
                ),
                "end": self.format_time(
                    self.get_value(
                        record,
                        "cctv_end_time",
                        "end_timestamp",
                        "end_time",
                        default="",
                    )
                ),
                "elapsed_start": elapsed_start,
            })

        intervals.sort(key=lambda item: item["elapsed_start"])

        activity_rows = {
            "active": [],
            "inactive": [],
            "not_visible": [],
        }

        for interval in intervals:
            activity_rows[interval["state"]].append(interval)

        max_records = max(
            len(activity_rows["active"]),
            len(activity_rows["inactive"]),
            len(activity_rows["not_visible"]),
        )

        for index in range(max_records):
            row = []

            for state in ("active", "inactive", "not_visible"):
                records = activity_rows[state]

                if index < len(records):
                    record = records[index]
                    row.extend([
                        index + 1,
                        record["start"],
                        record["end"],
                    ])
                else:
                    row.extend(["", "", ""])

            sheet.append(row)

        for row in sheet.iter_rows(min_row=5, max_col=9):
            for cell in row:
                cell.font = Font(size=8)
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

            sheet.row_dimensions[row[0].row].height = 18

        for column in range(1, 10):
            letter = get_column_letter(column)

            if (column - 1) % 3 == 0:
                sheet.column_dimensions[letter].width = 7
            else:
                sheet.column_dimensions[letter].width = 14

        last_row = max(4, sheet.max_row)

        self.apply_borders(sheet, f"A3:I{last_row}")

        sheet.freeze_panes = "A5"
        sheet.auto_filter.ref = f"A4:I{last_row}"

        self.configure_print(
            sheet,
            f"A1:I{last_row}",
            header_rows="1:4",
        )

    # --------------------------------------------------
    # Eating Frequency helpers
    # --------------------------------------------------

    def build_frequency_rows(self, frequency_records, eating_records):
        """
        Build the frequency table.

        Supports both:
        - Frequency
        - Frequency (Time)

        If the frequency CSV has no positive counts, use the eating-event
        CSV to count recorded eating events instead.
        """

        rows = []

        for record in frequency_records:
            duration = self.get_value(
                record,
                "Duration",
                "Time",
                "Time Range",
                default="",
            )

            behavior = str(
                self.get_value(
                    record,
                    "Behavior",
                    default="Eating",
                )
            ).strip().lower()

            if behavior and behavior != "eating":
                continue

            raw_frequency = self.get_value(
                record,
                "Frequency",
                "Frequency (Time)",
                "Count",
                "Occurrences",
                default=0,
            )

            try:
                frequency = int(float(raw_frequency or 0))
            except (TypeError, ValueError):
                frequency = 0

            rows.append({
                "duration": duration,
                "frequency": frequency,
            })

        # Use the frequency CSV when it contains positive counts.
        if rows and sum(row["frequency"] for row in rows) > 0:
            return rows

        # Otherwise, count the individual eating-event records.
        if not eating_records:
            return rows

        counts = defaultdict(int)

        for record in eating_records:
            behavior = str(
                self.get_value(
                    record,
                    "Behavior",
                    "behavior",
                    default="Eating",
                )
            ).strip().lower()

            if behavior and behavior != "eating":
                continue

            # Prefer an explicit hourly/time bucket if one exists.
            duration_bucket = self.get_value(
                record,
                "Duration",
                "Time Range",
                "Hour",
                "Time",
                default=None,
            )

            if duration_bucket:
                bucket = str(duration_bucket).strip()
            else:
                # Otherwise use the event's start timestamp.
                start_value = self.get_value(
                    record,
                    "Start Time",
                    "start_time",
                    "CCTV Start Time",
                    "start_timestamp",
                    default=None,
                )

                bucket = self.get_hour_bucket(start_value)

            counts[bucket] += 1

        if counts:
            return [
                {
                    "duration": bucket,
                    "frequency": count,
                }
                for bucket, count in sorted(counts.items())
            ]

        return rows

    @staticmethod
    def get_hour_bucket(value):
        """Convert a timestamp into an hourly label where possible."""
        if value is None or str(value).strip() == "":
            return "Unspecified"

        value = str(value).strip()

        # Try common timestamp formats.
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%H:%M:%S",
            "%H:%M",
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                hour = parsed.hour
                return f"{hour:02d}:00–{(hour + 1) % 24:02d}:00"
            except ValueError:
                continue

        return value

    # --------------------------------------------------
    # Eating Frequency
    # --------------------------------------------------

    def add_eating_sheet(
        self,
        workbook,
        frequency_records,
        eating_records,
    ):
        sheet = workbook.create_sheet("Eating Frequency")

        self.style_title(
            sheet,
            "A1:B1",
            "Eating Frequency by Hour",
        )

        sheet.merge_cells("A3:A4")

        sheet["A3"] = "Duration"
        sheet["B3"] = "Frequency"
        sheet["B4"] = "Eating"

        self.style_header(sheet["A3"])
        self.style_header(sheet["B3"])
        self.style_subheader(sheet["B4"], "70AD47")

        sheet.row_dimensions[3].height = 25
        sheet.row_dimensions[4].height = 25

        frequency_rows = self.build_frequency_rows(
            frequency_records,
            eating_records,
        )

        # Write data beginning at row 5.
        for record in frequency_rows:
            sheet.append([
                record["duration"],
                record["frequency"],
            ])

        sheet.column_dimensions["A"].width = 24
        sheet.column_dimensions["B"].width = 18

        for row in sheet.iter_rows(min_row=5, max_col=2):
            for cell in row:
                cell.font = Font(size=9)
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

            sheet.row_dimensions[row[0].row].height = 20

        last_row = max(4, sheet.max_row)

        self.apply_borders(sheet, f"A3:B{last_row}")

        sheet.freeze_panes = "A5"

        self.configure_print(
            sheet,
            f"A1:B{last_row}",
            header_rows="1:4",
        )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    def add_visualization_sheet(
        self,
        workbook,
        activity_records,
        eating_records,
        metadata_records,
    ):
        sheet = workbook.create_sheet("Visualization")

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        self.style_title(
            sheet,
            "A1:F1",
            "Civet Monitoring Overview",
        )

        sheet.row_dimensions[1].height = 35

        # --------------------------------------------------
        # Recording information
        # --------------------------------------------------

        metadata = {
            row.get("metadata"): row.get("value")
            for row in metadata_records
        }

        sheet.merge_cells("A3:F3")
        sheet["A3"] = "RECORDING INFORMATION"
        sheet["A3"].font = Font(
            bold=True,
            size=11,
            color="1F4E78",
        )
        sheet["A3"].alignment = Alignment(
            vertical="center",
            horizontal="left",
        )
        sheet.row_dimensions[3].height = 25

        information = [
            (
                "A4:B4",
                "A5:B5",
                "RECORDING DATE",
                metadata.get("recording_date") or "Unavailable",
            ),
            (
                "C4:D4",
                "C5:D5",
                "RECORDING START",
                metadata.get("recording_start_timestamp") or "Unavailable",
            ),
            (
                "E4:F4",
                "E5:F5",
                "RECORDING END",
                metadata.get("recording_end_timestamp") or "Unavailable",
            ),
        ]

        for label_range, value_range, label, value in information:
            sheet.merge_cells(label_range)
            sheet.merge_cells(value_range)

            label_cell = sheet[label_range.split(":")[0]]
            label_cell.value = label
            label_cell.fill = PatternFill(
                "solid",
                fgColor="D9EAF7",
            )
            label_cell.font = Font(
                bold=True,
                size=8,
                color="45627C",
            )
            label_cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
            )

            value_cell = sheet[value_range.split(":")[0]]
            value_cell.value = value
            value_cell.fill = PatternFill(
                "solid",
                fgColor="EAF2F8",
            )
            value_cell.font = Font(
                bold=True,
                size=10,
                color="20364A",
            )
            value_cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

            self.apply_borders(sheet, label_range)
            self.apply_borders(sheet, value_range)

        sheet.row_dimensions[4].height = 20
        sheet.row_dimensions[5].height = 30

        # Video duration.
        video_duration_seconds = self.parse_duration(
            metadata.get("video_duration_seconds")
        )

        video_duration = self.format_elapsed(video_duration_seconds)

        sheet.merge_cells("A6:F6")
        sheet["A6"] = f"VIDEO DURATION: {video_duration}"
        sheet["A6"].font = Font(
            bold=True,
            size=10,
            color="20364A",
        )
        sheet["A6"].alignment = Alignment(
            horizontal="center",
            vertical="center",
        )
        sheet["A6"].fill = PatternFill(
            "solid",
            fgColor="D9EAF7",
        )

        self.apply_borders(sheet, "A6:F6")
        sheet.row_dimensions[6].height = 25

        # --------------------------------------------------
        # Calculate activity durations
        # --------------------------------------------------

        durations = defaultdict(float)

        for record in activity_records:
            state = str(record.get("state", "")).strip().lower()

            if state in {"active", "inactive", "not_visible"}:
                durations[state] += self.get_record_duration(record)

        active = durations["active"]
        inactive = durations["inactive"]
        not_visible = durations["not_visible"]

        total_activity = active + inactive + not_visible

        total_duration = video_duration_seconds

        if total_duration <= 0:
            total_duration = total_activity

        # --------------------------------------------------
        # Activity summary heading
        # --------------------------------------------------

        sheet.merge_cells("A8:F8")
        sheet["A8"] = "ACTIVITY SUMMARY"
        sheet["A8"].font = Font(
            bold=True,
            size=11,
            color="1F4E78",
        )
        sheet["A8"].alignment = Alignment(
            horizontal="left",
            vertical="center",
        )
        sheet.row_dimensions[8].height = 25

        # --------------------------------------------------
        # Activity KPI cards
        # --------------------------------------------------

        activity_cards = [
            {
                "columns": ("A", "B"),
                "label": "ACTIVE",
                "duration": active,
                "color": "FCEBDD",
                "text_color": "9B541E",
            },
            {
                "columns": ("C", "D"),
                "label": "INACTIVE",
                "duration": inactive,
                "color": "E8F1E5",
                "text_color": "47723D",
            },
            {
                "columns": ("E", "F"),
                "label": "NOT VISIBLE",
                "duration": not_visible,
                "color": "EEEEEE",
                "text_color": "666666",
            },
        ]

        for card in activity_cards:
            first, last = card["columns"]

            label_range = f"{first}9:{last}9"
            duration_range = f"{first}10:{last}10"
            percentage_range = f"{first}11:{last}11"

            percentage = (
                card["duration"] / total_duration * 100
                if total_duration > 0
                else 0
            )

            sheet.merge_cells(label_range)
            sheet.merge_cells(duration_range)
            sheet.merge_cells(percentage_range)

            label_cell = sheet[f"{first}9"]
            label_cell.value = card["label"]

            duration_cell = sheet[f"{first}10"]
            duration_cell.value = self.format_elapsed(card["duration"])

            percentage_cell = sheet[f"{first}11"]
            percentage_cell.value = f"{percentage:.1f}%"

            for cell in (
                label_cell,
                duration_cell,
                percentage_cell,
            ):
                cell.fill = PatternFill(
                    "solid",
                    fgColor=card["color"],
                )
                cell.font = Font(
                    color=card["text_color"],
                    bold=True,
                )
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

            label_cell.font = Font(
                color=card["text_color"],
                bold=True,
                size=9,
            )

            duration_cell.font = Font(
                color=card["text_color"],
                bold=True,
                size=16,
            )

            percentage_cell.font = Font(
                color=card["text_color"],
                size=10,
            )

            self.apply_borders(sheet, label_range)
            self.apply_borders(sheet, duration_range)
            self.apply_borders(sheet, percentage_range)

        sheet.row_dimensions[9].height = 22
        sheet.row_dimensions[10].height = 35
        sheet.row_dimensions[11].height = 23

        # --------------------------------------------------
        # Activity distribution chart
        # --------------------------------------------------

        sheet.merge_cells("A13:F13")
        sheet["A13"] = "ACTIVITY DISTRIBUTION"
        sheet["A13"].font = Font(
            bold=True,
            size=11,
            color="1F4E78",
        )
        sheet["A13"].alignment = Alignment(
            horizontal="left",
            vertical="center",
        )
        sheet.row_dimensions[13].height = 25

        # Keep chart source data outside the print area.
        sheet["H1"] = "Activity"
        sheet["I1"] = "Duration (seconds)"

        sheet["H2"] = "Active"
        sheet["I2"] = active

        sheet["H3"] = "Inactive"
        sheet["I3"] = inactive

        # Hide the source columns but explicitly allow the chart
        # to display data from hidden cells.
        sheet.column_dimensions["H"].hidden = True
        sheet.column_dimensions["I"].hidden = True

        chart = PieChart()
        chart.title = "Active vs Inactive Time"
        chart.style = 10
        chart.height = 8
        chart.width = 13
        chart.varyColors = True

        data = Reference(
            sheet,
            min_col=9,
            min_row=1,
            max_row=3,
        )

        labels = Reference(
            sheet,
            min_col=8,
            min_row=2,
            max_row=3,
        )

        chart.add_data(data, titles_from_data=True)
        chart.set_categories(labels)

        chart.dataLabels = DataLabelList()
        chart.dataLabels.showPercent = True
        chart.dataLabels.showVal = False
        chart.dataLabels.showLeaderLines = True

        # Important: plot data even when its source columns are hidden.
        chart.visible_cells_only = False

        # Do not create a misleading chart if both values are zero.
        if active > 0 or inactive > 0:
            sheet.add_chart(chart, "A14")
        else:
            sheet.merge_cells("A15:F15")
            sheet["A15"] = "No active or inactive duration data available."
            sheet["A15"].alignment = Alignment(
                horizontal="center",
                vertical="center",
            )
            sheet["A15"].font = Font(
                italic=True,
                color="808080",
                size=9,
            )

        # --------------------------------------------------
        # Layout and formatting
        # --------------------------------------------------

        for column in "ABCDEF":
            sheet.column_dimensions[column].width = 17

        for row in sheet.iter_rows(
            min_row=1,
            max_row=30,
            min_col=1,
            max_col=6,
        ):
            for cell in row:
                if cell.value is not None:
                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center",
                        wrap_text=True,
                    )

        # Restore section heading alignment.
        for cell in ("A3", "A8", "A13"):
            sheet[cell].alignment = Alignment(
                horizontal="left",
                vertical="center",
            )

        self.configure_print(sheet, "A1:F30")

    # --------------------------------------------------
    # Generate report
    # --------------------------------------------------

    def generate(
        self,
        activity_records,
        eating_csv_path,
        frequency_csv_path,
        metadata_csv_path,
    ):
        """Create the final Excel workbook."""

        eating_records = self.read_csv(eating_csv_path)
        frequency_records = self.read_csv(frequency_csv_path)
        metadata_records = self.read_csv(metadata_csv_path)

        workbook = Workbook()

        self.add_activity_sheet(
            workbook,
            activity_records,
        )

        self.add_eating_sheet(
            workbook,
            frequency_records,
            eating_records,
        )

        self.add_visualization_sheet(
            workbook,
            activity_records,
            eating_records,
            metadata_records,
        )

        workbook.save(self.report_path)

        print(f"Excel report: {self.report_path}")

        return self.report_path
