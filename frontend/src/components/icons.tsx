import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const baseProps = {
  width: 22,
  height: 22,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function HomeIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M3.2 11.3 12 3.6l8.9 7.8" />
      <path d="M5 10.8v9.4c0 .3.2.5.5.5h4.2v-5.4c0-.3.2-.6.5-.6h3.6c.3 0 .6.2.6.6v5.4h4.1c.3 0 .5-.3.5-.6v-9.3" />
    </svg>
  );
}

export function ShortsIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M13.8 3.2c-2 .2-3.9 1.2-5.6 2.3-1.7 1-3.4 2.4-3.8 4.2-.4 1.7.8 3 2.4 3.4.5.1 1 .2 1.6.4" />
      <path d="M10.2 20.8c2-.2 3.9-1.2 5.6-2.3 1.7-1 3.4-2.4 3.8-4.2.4-1.7-.8-3-2.4-3.4-.5-.1-1-.2-1.6-.4" />
      <path d="M10.2 9.4 14.7 12l-4.5 2.6V9.4Z" />
    </svg>
  );
}

export function SubsIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M5 6h14" />
      <path d="M6.5 9.5h11" />
      <path d="M4.5 13c-.1 2 .2 5.2 1.4 6.2 1.3 1 10.9 1 12.2 0 1.2-1 1.5-4.1 1.4-6.2-.1-2-.5-4-1.4-4.7-1.3-1-10.9-1-12.2 0-.9.7-1.3 2.7-1.4 4.7Z" />
      <path d="M11 13.5 14.3 16 11 18Z" />
    </svg>
  );
}

export function LibraryIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M4.5 5.2c0-.4.3-.7.7-.6l3 .4c.3 0 .5.3.5.6V19c0 .4-.3.6-.6.6l-3-.4c-.3 0-.6-.3-.6-.6V5.2Z" />
      <path d="M9.8 5.6c0-.4.3-.6.6-.6h2.8c.3 0 .6.3.6.6V19c0 .3-.3.6-.6.6h-2.8c-.4 0-.6-.3-.6-.6V5.6Z" />
      <path d="m15 5.9.4-.1c.3-.1.6.1.7.4L19 18.4c.1.3-.1.6-.4.7l-2.7.7c-.3.1-.6-.1-.7-.4L12 7c-.1-.3.1-.6.4-.7l2.6-.4Z" />
    </svg>
  );
}

export function HistoryIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M3.6 9.2c1-3.4 4.4-5.8 8.2-5.7 4.6 0 8.5 3.7 8.6 8.3.1 4.7-3.6 8.7-8.4 8.9-3.4.1-6.5-1.7-7.9-4.6" />
      <path d="M3.4 4.6 3.6 9.4l4.7-.4" />
      <path d="M12 7.6V12l3.2 2.1" />
    </svg>
  );
}

export function YourVideosIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M3.6 8.6c.1-.6.6-1 1.2-1l14 .2c.6 0 1 .5 1 1l-.2 9.4c0 .6-.5 1-1.1 1l-14-.2c-.6 0-1-.5-1-1l.1-9.4Z" />
      <path d="M3.9 8 5.6 5l3.5.1L7.4 8" />
      <path d="m8.9 7.9 1.8-3 3.5.2-1.8 2.9" />
      <path d="m13.4 8 2-3 3.4.1L17 8" />
      <path d="m10.8 11.3 4.6 2.7-4.6 2.7v-5.4Z" />
    </svg>
  );
}

export function WatchIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M2.5 12c1.4-3.3 5.3-6 9.5-6 4.3 0 8.1 2.7 9.5 6-1.4 3.3-5.2 6-9.5 6-4.2 0-8.1-2.7-9.5-6Z" />
      <path d="M12 8.6c1.9 0 3.4 1.5 3.4 3.4 0 1.9-1.5 3.4-3.4 3.4-1.9 0-3.4-1.5-3.4-3.4 0-1.9 1.5-3.4 3.4-3.4Z" />
      <path d="M12 10.3c.9 0 1.7.7 1.7 1.7" />
    </svg>
  );
}

export function HubIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M4 4.4v15.2" />
      <path d="M4 19.6h16" />
      <path d="M7.5 16V11" />
      <path d="M11.2 16V7" />
      <path d="M14.8 16v-6" />
      <path d="M18.5 16V8" />
    </svg>
  );
}

export function SettingsIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M11 3.6c-.3 0-.6.2-.7.5l-.4 1.7c-.6.2-1.2.4-1.7.7L6.6 5.5c-.3-.2-.7-.1-.9.1L4.4 7.1c-.2.2-.3.6-.1.9l1 1.5c-.4.5-.6 1.1-.8 1.7l-1.7.4c-.3 0-.5.4-.5.7v1.9c0 .3.2.6.5.7l1.7.4c.2.6.4 1.2.7 1.7l-1 1.5c-.2.3-.1.7.1.9l1.3 1.4c.2.2.6.3.9.1l1.6-1c.5.3 1.1.6 1.7.7l.4 1.7c0 .3.4.5.7.5h1.9c.3 0 .6-.2.7-.5l.4-1.7c.6-.2 1.2-.4 1.7-.7l1.6 1c.3.2.7.1.9-.1l1.4-1.4c.2-.2.3-.6.1-.9l-1-1.5c.3-.5.6-1.1.7-1.7l1.7-.4c.3 0 .5-.4.5-.7v-1.9c0-.3-.2-.6-.5-.7l-1.7-.4c-.2-.6-.4-1.2-.7-1.7l1-1.5c.2-.3.1-.7-.1-.9L18 5.6c-.2-.2-.6-.3-.9-.1l-1.5 1c-.5-.3-1.1-.6-1.7-.7l-.4-1.7c0-.3-.4-.5-.7-.5H11Z" />
      <path d="M12 9.4c1.5 0 2.7 1.2 2.7 2.7 0 1.5-1.2 2.7-2.7 2.7-1.5 0-2.7-1.2-2.7-2.7 0-1.5 1.2-2.7 2.7-2.7Z" />
    </svg>
  );
}

export function ShieldIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M12 3.4c-.1 0-.3 0-.4.1l-6.5 2.2c-.3.1-.5.4-.5.7 0 4.4.6 11 7 13.9.1.1.3.1.4.1.2 0 .3 0 .4-.1 6.4-2.9 7-9.5 7-13.9 0-.3-.2-.6-.5-.7l-6.5-2.2c-.1-.1-.3-.1-.4-.1Z" />
      <path d="m8.7 12 2.4 2.5 4.3-4.6" />
    </svg>
  );
}

export function MenuIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M4 6.3c4.7-.2 11.5-.2 16 .1" />
      <path d="M4 12c5-.3 11.5-.3 16 0" />
      <path d="M4 17.6c4.6-.2 11.4-.2 16 .1" />
    </svg>
  );
}

export function SearchIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M10.7 4.2c3.6 0 6.6 3 6.5 6.6 0 1.6-.6 3-1.6 4.1" />
      <path d="M15.6 14.9c-1.2 1.2-2.9 1.9-4.7 1.9-3.7 0-6.7-2.9-6.7-6.5 0-1.7.7-3.3 1.8-4.5" />
      <path d="m15.7 15.7 4.4 4.2" />
    </svg>
  );
}

export function GlobeIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M3.5 12c0-4.7 3.8-8.5 8.5-8.5 4.6 0 8.4 3.8 8.5 8.4 0 4.7-3.8 8.6-8.5 8.6-4.7 0-8.5-3.8-8.5-8.5Z" />
      <path d="M3.7 11.6c5.5.3 11.1.3 16.6 0" />
      <path d="M12 3.6c2.5 2.4 3.9 5.4 3.9 8.4 0 3-1.4 6-3.9 8.4" />
      <path d="M12 3.6c-2.5 2.4-3.9 5.4-3.9 8.4 0 3 1.4 6 3.9 8.4" />
    </svg>
  );
}

export function SunIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M12 7.7c2.4 0 4.3 1.9 4.3 4.3 0 2.4-1.9 4.3-4.3 4.3-2.4 0-4.3-1.9-4.3-4.3 0-2.4 1.9-4.3 4.3-4.3Z" />
      <path d="M12 2.7v2" />
      <path d="M12 19.3v2" />
      <path d="m4.8 4.8 1.4 1.4" />
      <path d="m17.8 17.8 1.4 1.4" />
      <path d="M2.7 12h2" />
      <path d="M19.3 12h2" />
      <path d="m4.8 19.2 1.4-1.4" />
      <path d="m17.8 6.2 1.4-1.4" />
    </svg>
  );
}

export function MoonIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M20.3 13.5c-.9 3.6-4.2 6.3-8 6.3-4.6 0-8.3-3.7-8.3-8.3 0-3.9 2.8-7.2 6.6-8 .2 0 .4.1.4.3-.4 1-.6 2.1-.6 3.2 0 4.6 3.7 8.3 8.3 8.3.4 0 .9 0 1.3-.1.2 0 .4.2.3.3Z" />
    </svg>
  );
}

export function DatabaseIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M4.4 6c0-1.4 3.4-2.5 7.6-2.5S19.6 4.6 19.6 6c0 1.4-3.4 2.5-7.6 2.5S4.4 7.4 4.4 6Z" />
      <path d="M4.4 6v12c0 1.4 3.4 2.5 7.6 2.5s7.6-1.1 7.6-2.5V6" />
      <path d="M4.4 12c0 1.4 3.4 2.5 7.6 2.5s7.6-1.1 7.6-2.5" />
    </svg>
  );
}

export function PlayIcon(p: IconProps) {
  return (
    <svg {...baseProps} {...p}>
      <path d="M8 5.4v13.2c0 .5.5.7.9.5l11.2-6.6c.4-.2.4-.8 0-1L8.9 4.9c-.4-.2-.9 0-.9.5Z" />
    </svg>
  );
}
