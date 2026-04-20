//  1D Grain proof of concept
//  built upon MFEM examples 16, 22, 39

/* The solution should look like such:
 *
 * [K] [T] = [Q] + [q]
 *
 * Where:
 *  K is the conductivity matrix
 *  T is the nodal temperature
 *  Q is the thermal load from heat source
 *  q is the vector of nodal heat flow across the cross-section
 *
 * This solution should have both a real and imaginary part.
 *
 */

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/fe_coll.hpp>
#include <mfem/fem/fespace.hpp>
#include <mfem/fem/gridfunc.hpp>
#include <mfem/linalg/complex_operator.hpp>
#include <mfem/linalg/operator.hpp>
#include <mfem/linalg/vector.hpp>
#include <mfem/mesh/attribute_sets.hpp>

using namespace std;
using namespace mfem;

// Material Properties
static real_t c_p = 1.6e6; // Heat Capacity [J/m^3-K]
static real_t k = 1.3;     // Thermal conductivity [W/m-K]
static real_t rho = 2200;  // Density of SiO_2 [Kg/m^3]
static real_t thermalDiffusivity =
    k / (c_p * rho); // Thermal diffusivity of material properties [m^2/s]

class ConductionOperator : public TimeDependentOperator {
protected:
  FiniteElementSpace &fespace;
  Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

  BilinearForm *M;
  BilinearForm *K;

  SparseMatrix Mmat, Kmat;
  SparseMatrix *T; // T = M + dt K
  real_t current_dt;

  CGSolver M_solver; // Krylov solver for inverting the mass matrix M
  DSmoother M_prec;  // Preconditioner for the mass matrix M

  CGSolver T_solver; // Implicit solver for T = M + dt K
  DSmoother T_prec;  // Preconditioner for the implicit solver

  real_t alpha, kappa;

  mutable Vector z; // auxiliary vector

public:
  ConductionOperator(FiniteElementSpace &f, real_t alpha, real_t kappa,
                     const Vector &u);

  void Mult(const Vector &u, Vector &du_dt) const override;
  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

  /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
  void SetParameters(const Vector &u);

  ~ConductionOperator() override;
};

class ThermalOperator : public ComplexOperator {
protected:
  FiniteElementSpace &fespace;
  Array<int> ess_tdof_list; // essential boundary conditon list (dirichlet)

  BilinearForm *M;
  BilinearForm *K;

  SparseMatrix Mmat, Kmat;
  SparseMatrix *T; // T = M + dt K

  real_t alpha;

  mutable Vector z; // auxiliary vector

public:
  ThermalOperator(FiniteElementSpace &f, real_t alpha,
                  Convention convention = HERMITIAN);
};

real_t InitialTemperature(const Vector &x);
real_t omega_ = 10.0;

int main(int argc, char *argv[]) {
  // Parameters for the analysis.
  const char *mesh_file = "2D_Grain.msh";
  int order = 2;
  int ref_levels = 2;

  int ode_solver_type = 23; // SDIRK33Solver
  real_t t_final = 0.5;
  // real_t t_final = 0.3;

  real_t dt = 1.0e-2;
  real_t alpha = 1.0e-2;
  // real_t kappa = 0.5;
  real_t kappa = 0.1;

  bool visualization = true;
  bool visit = false;
  int vis_steps = 5;
  bool solve_implicit_state = false;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 ODESolver::Types.c_str());
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step", "Time step.");
  args.AddOption(&alpha, "-a", "--alpha", "Alpha coefficient.");
  args.AddOption(&kappa, "-k", "--kappa", "Kappa coefficient offset.");
  args.AddOption(&solve_implicit_state, "-imp-state", "--implicit-state",
                 "-imp-slope", "--implicit-slope",
                 "Implicitly solve for stage state or slope.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                 "--no-visit-datafiles",
                 "Save data files for VisIt (visit.llnl.gov) visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 3. Define the ODE solver used for time integration. Several implicit
  //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
  //    explicit Runge-Kutta methods are available.
  unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  AttributeSets &attr_sets = mesh->attribute_sets;
  AttributeSets &bdr_attr_sets = mesh->bdr_attribute_sets;
  {
    std::set<string> names = attr_sets.GetAttributeSetNames();
    cout << "Element Attribute Set Names: ";
    for (auto const &set_name : names) {
      cout << " \"" << set_name << "\"";
    }
    cout << endl;

    std::set<string> bdr_names = attr_sets.GetAttributeSetNames();
    cout << "Boundary Attribute Set Names: ";
    for (auto const &bdr_set_name : bdr_names) {
      cout << " \"" << bdr_set_name << "\"";
    }
    cout << endl;
  }

  // 5. Define the vector finite element space representing the current and the
  //    initial temperature, u_ref.
  H1_FECollection fe_coll(order, dim);
  FiniteElementSpace fespace(mesh, &fe_coll);

  int fe_size = fespace.GetTrueVSize();
  cout << "Number of temperature unknowns: " << fe_size << endl;

  GridFunction u_gf(&fespace);

  // 6. Set the initial conditions for u. All boundaries are considered
  //    natural.
  FunctionCoefficient u_0(InitialTemperature);
  u_gf.ProjectCoefficient(u_0);
  Vector u;
  u_gf.GetTrueDofs(u);

  // 7. Initialize the conduction operator and the visualization.
  // ConductionOperator oper(fespace, alpha, kappa, u);
  ConductionOperator oper(fespace, thermalDiffusivity, kappa, u);
  using ImplicitVariableType = ConductionOperator::ImplicitVariableType;
  ImplicitVariableType imp_var = solve_implicit_state
                                     ? ImplicitVariableType::STATE
                                     : ImplicitVariableType::SLOPE;
  oper.SetImplicitVariableType(imp_var);

  u_gf.SetFromTrueDofs(u);
  {
    ofstream omesh("ex16.mesh");
    omesh.precision(precision);
    mesh->Print(omesh);
    ofstream osol("ex16-init.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  VisItDataCollection visit_dc("Example16", mesh);
  visit_dc.RegisterField("temperature", &u_gf);
  if (visit) {
    visit_dc.SetCycle(0);
    visit_dc.SetTime(0.0);
    visit_dc.Save();
  }

  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    sout.open(vishost, visport);
    if (!sout) {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      visualization = false;
      cout << "GLVis visualization disabled.\n";
    } else {
      sout.precision(precision);
      sout << "solution\n" << *mesh << u_gf;
      sout << "pause\n";
      sout << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
    }
  }

  // 8. Perform time-integration (looping over the time iterations, ti, with a
  //    time-step dt).
  ode_solver->Init(oper);
  real_t t = 0.0;

  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    if (t + dt >= t_final - dt / 2) {
      last_step = true;
    }

    ode_solver->Step(u, t, dt);

    if (last_step || (ti % vis_steps) == 0) {
      cout << "step " << ti << ", t = " << t << endl;

      u_gf.SetFromTrueDofs(u);
      if (visualization) {
        sout << "solution\n" << *mesh << u_gf << flush;
      }

      if (visit) {
        visit_dc.SetCycle(ti);
        visit_dc.SetTime(t);
        visit_dc.Save();
      }
    }
    oper.SetParameters(u);
  }

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m ex16.mesh -g ex16-final.gf".
  {
    ofstream osol("ex16-final.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  // 10. Free the used memory.
  delete mesh;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, real_t al,
                                       real_t kap, const Vector &u)
    : TimeDependentOperator(f.GetTrueVSize(), (real_t)0.0), fespace(f), M(NULL),
      K(NULL), T(NULL), current_dt(0.0), z(height) {
  const real_t rel_tol = 1e-8;

  M = new BilinearForm(&fespace);
  M->AddDomainIntegrator(new MassIntegrator());
  M->Assemble();
  M->FormSystemMatrix(ess_tdof_list, Mmat);

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(30);
  M_solver.SetPrintLevel(0);
  M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(Mmat);

  alpha = al;
  kappa = kap;

  T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(100);
  T_solver.SetPrintLevel(0);
  T_solver.SetPreconditioner(T_prec);

  SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const {
  // Compute:
  //    du_dt = M^{-1}*-Ku
  // for du_dt, where K is linearized by using u from the previous timestep
  Kmat.Mult(u, z);
  z.Neg(); // z = -z
  M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const real_t dt, const Vector &u,
                                       Vector &k) {
  // Solve the equation:
  //    M*k = -K(u + dt*k) for k = du/dt, if solving for stage-slope
  // or
  //    M*k = -dt*K(k) + M*u for k = u_s, if solving for stage-state
  // where K is linearized by using u from the previous timestep, and
  // the stage-state and slope relation: du/dt = (u_s - u)/dt.
  if (!T) {
    T = Add(1.0, Mmat, dt, Kmat);
    current_dt = dt;
    T_solver.SetOperator(*T);
  }
  MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

  // Construct current right-hand side for stage state vs. slope solve
  if (ImplicitVarTypeIsState()) {
    // k, on return, is the stage value u_s
    Mmat.Mult(u, z);
  } else {
    // k, on return, is the stage slope du/dt
    Kmat.Mult(u, z);
    z.Neg();
  }
  T_solver.Mult(z, k);
}

void ConductionOperator::SetParameters(const Vector &u) {
  GridFunction u_alpha_gf(&fespace);
  u_alpha_gf.SetFromTrueDofs(u);
  for (int i = 0; i < u_alpha_gf.Size(); i++) {
    u_alpha_gf(i) = kappa + alpha * u_alpha_gf(i);
  }

  delete K;
  K = new BilinearForm(&fespace);

  GridFunctionCoefficient u_coeff(&u_alpha_gf);

  K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
  K->Assemble();
  K->FormSystemMatrix(ess_tdof_list, Kmat);
  delete T;
  T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator() {
  delete T;
  delete M;
  delete K;
}

real_t InitialTemperature(const Vector &x) {
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}

real_t InitialFlux(const Vector &x) { return 1.0; }
