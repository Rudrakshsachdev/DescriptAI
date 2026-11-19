export interface ProductTemplate {
  id: string;
  name: string;
  shortLabel: string;
  category: string;
  brand: string;
  iconName: "Lamp" | "Watch" | "Droplets" | "Headphones" | "Briefcase" | "Coffee";
  features: string;
  specifications: string;
  audience: string;
  defaultTone?: "Refined" | "Friendly" | "Confident" | "Playful";
  defaultLength?: "Short" | "Standard" | "Detailed";
}

export const PRODUCT_TEMPLATES: ProductTemplate[] = [
  {
    id: "smartwatch",
    name: "ApexPro GPS Smartwatch",
    shortLabel: "Smartwatch",
    category: "Electronics & Tech",
    brand: "Veloce Active",
    iconName: "Watch",
    features: `Dual-frequency precision GPS tracking with offline topographic maps
Continuous 24/7 heart rate, HRV, and SpO2 blood oxygen monitoring
14-day extended battery life in smartwatch mode (36 hours in GPS mode)
5 ATM water resistance rated for open-water swimming and surfing
Titanium bezel with scratch-resistant sapphire crystal display`,
    specifications: `Display: 1.4-inch AMOLED, 454 × 454 resolution (1,000 nits)
Case Material: Grade 5 Titanium and fiber-reinforced polymer
Battery: 450 mAh Lithium-Polymer, magnetic fast charging
Connectivity: Bluetooth 5.3, dual-band Wi-Fi, NFC contactless pay
Sensors: Barometric altimeter, 3-axis compass, optical HR sensor
Weight: 48 g (without strap)`,
    audience: "Marathon runners, trail athletes, and fitness enthusiasts tracking performance",
    defaultTone: "Confident",
    defaultLength: "Detailed",
  },
  {
    id: "skincare-serum",
    name: "HydraGlow Botanical Renewal Serum",
    shortLabel: "Skincare Serum",
    category: "Beauty & Personal Care",
    brand: "Aura Botanicals",
    iconName: "Droplets",
    features: `Triple-molecular weight hyaluronic acid for deep multi-depth hydration
Cold-pressed organic rosehip seed oil and plant squalane barrier support
Enriched with 5% Niacinamide and stabilized Vitamin C to brighten skin tone
Fast-absorbing, non-comedogenic lightweight emulsion without sticky residue
Clean formula: 100% vegan, fragrance-free, paraben-free, and cruelty-free`,
    specifications: `Volume: 30 mL (1.0 fl oz) dropper bottle
Skin Type: Suitable for sensitive, dry, and combination skin types
pH Level: Skin-balanced 5.5
Packaging: UV-protective amber glass bottle with precision pipette
Certifications: Leaping Bunny Certified, Dermatologist Tested`,
    audience: "Individuals with sensitive or dry skin seeking restored moisture and a radiant glow",
    defaultTone: "Refined",
    defaultLength: "Standard",
  },
  {
    id: "wireless-headphones",
    name: "SonicWave ANC Wireless Headphones",
    shortLabel: "Headphones",
    category: "Electronics & Tech",
    brand: "AcoustiQ Audio",
    iconName: "Headphones",
    features: `Adaptive hybrid active noise cancellation with natural ambient transparency
Custom 40mm bio-cellulose drivers tuned for audiophile clarity and punchy bass
45-hour continuous battery playback with USB-C quick charge (5 hrs from 10 min charge)
Multipoint Bluetooth pairing for seamless switching between laptop and smartphone
Ultra-soft memory foam ear cushions covered in breathable protein leather`,
    specifications: `Bluetooth Version: 5.3 with LDAC, AAC, and SBC audio codecs
Frequency Response: 20 Hz – 40,000 Hz (Hi-Res Audio Certified)
Microphones: 4 beamforming mics with AI environmental noise suppression
Charging: USB-C fast charging, 3.5mm analog audio cable included
Weight: 245 g`,
    audience: "Remote workers, commuters, and music lovers seeking active noise cancellation",
    defaultTone: "Refined",
    defaultLength: "Detailed",
  },
  {
    id: "leather-backpack",
    name: "Vanguard Full-Grain Leather Commuter Backpack",
    shortLabel: "Leather Backpack",
    category: "Apparel & Fashion",
    brand: "Atelier & Co.",
    iconName: "Briefcase",
    features: `Handcrafted from vegetable-tanned full-grain Tuscan cowhide leather
Dedicated shock-absorbing padded laptop sleeve accommodating up to 16-inch laptops
Ergonomic contoured leather shoulder straps with breathable mesh back panel
Luggage trolley pass-through sleeve for effortless airport travel
Concealed anti-theft passport pocket and quick-access magnetic front compartment`,
    specifications: `Dimensions: 43 × 30 × 14 cm (18 L capacity)
Hardware: Solid antique brass hardware and heavy-duty YKK Excella zippers
Lining: Water-resistant recycled cotton herringbone weave
Compartments: Main zip chamber, laptop sleeve, 2 pen slots, key leash
Weight: 1.25 kg`,
    audience: "Urban professionals, executives, and commuters seeking handcrafted timeless design",
    defaultTone: "Confident",
    defaultLength: "Standard",
  },
  {
    id: "coffee-maker",
    name: "BaristaCraft Precision Espresso Machine",
    shortLabel: "Espresso Maker",
    category: "Kitchen & Dining",
    brand: "CafféForma",
    iconName: "Coffee",
    features: `15-bar Italian vibration pump delivering optimal pressure for rich golden crema
ThermoBlock instant heating system reaching brew temperature in under 35 seconds
Commercial-style 360-degree stainless steel steam wand for velvety microfoam latte art
Volumetric shot programming for single and double espresso extractions
Integrated cup warming tray and pre-infusion cycle for balanced espresso extraction`,
    specifications: `Water Reservoir: 1.8 L removable BPA-free tank with water level indicator
Portafilter: 58 mm commercial-grade brass portafilter with dual-wall baskets
Power: 1450 W, 120 V AC, auto shut-off after 20 minutes
Body Material: Brushed 304 food-grade stainless steel
Dimensions: 31 × 28 × 34 cm | Weight: 6.8 kg`,
    audience: "Home coffee enthusiasts and espresso lovers wanting authentic café-quality drinks",
    defaultTone: "Friendly",
    defaultLength: "Standard",
  },
  {
    id: "desk-lamp",
    name: "LumaArc Desk Lamp",
    shortLabel: "Desk Lamp",
    category: "Home & lighting",
    brand: "Northline Studio",
    iconName: "Lamp",
    features: `Adjustable warm-to-cool light with 5 color temperatures (2700K - 6500K)
Touch-sensitive step-less dimmer with last-setting memory function
Built-in 15W USB-C fast charging port for smartphones and accessories
Glare-free architectural light bar with asymmetric optical design`,
    specifications: `Materials: Brushed anodized aluminum and recycled matte ABS
Dimensions: 42 × 16 cm base, 45 cm maximum arm height
Power: 12 W LED, USB-C 5V/3A power input
Color Rendering Index: Ra > 95 for true color accuracy
Lifespan: 50,000 hours`,
    audience: "Design-conscious professionals and remote workers setting up a modern workspace",
    defaultTone: "Refined",
    defaultLength: "Standard",
  },
];
