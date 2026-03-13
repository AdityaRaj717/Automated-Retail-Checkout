import "./globals.css";

export const metadata = {
  title: "Retail Checkout System",
  description: "Automated retail checkout system with camera-based product detection",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
