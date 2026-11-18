import { GeneratorForm } from "@/components/generator/GeneratorForm";

export const metadata = {
  title: "Generate Descriptions — Descript AI",
  description: "Generate e-commerce descriptions with fine-tuned FLAN-T5 Seq2Seq model.",
};

export default function GeneratePage() {
  return (
    <div className="pb-12">
      <GeneratorForm />
    </div>
  );
}
